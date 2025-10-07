# Langgraphのチェックポイント、Human-in-the-Loop、ストリーミングを活用したエージェント
# AIエージェントはユーザーからのフィードバックを反映して、次の行動を決定します。

from botocore.config import Config
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    ToolCall
)
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

from dotenv import load_dotenv

load_dotenv()

# ツールの定義
# Web検索ツール
web_search = TavilySearch(max_results=2, topic="general")

working_dir = "report"

# ローカルファイルを扱うツールキット
file_toolkit = FileManagementToolkit(
    root_dir=str(working_dir),
    selected_tools=["write_file"],
    )

write_file = file_toolkit.get_tools()[0]

# 使用するツールのリスト
tools = [web_search, write_file]
tools_by_name = {tool.name: tool for tool in tools}

# LLMの初期化
cfg = Config(read_timeout=300)
llm_with_tools = init_chat_model(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_provider="bedrock_converse",
    config=cfg,
).bind_tools(tools)

# システムプロンプト
system_prompt = """
あなたの責務はユーザーからのリクエストを調査し、調査結果をファイルに出力することです。
- ユーザーのリクエスト調査にWeb検索が必要であれば、Web検索ツールを使用してください。
- 必要な情報が集まったと判断したら検索は終了して下さい。
- 検索は最大2回までにして下さい。
- ファイル形式はHTML形式に変換して保存して下さい。
  * Web検索が拒否された場合、Web検索を中止して下さい。
  * レポート保存が拒否された場合、レポート保存を中止し、内容をユーザーに伝えて下さい。
"""

# LLMを呼び出すタスク
@task
def invoke_llm(messages: list[BaseMessage]) -> AIMessage:
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    return response

# ツールを実行するタスク
@task
def use_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])

# ユーザーにツール実行の承認を求める
def ask_human(tool_call: ToolCall):
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_data = {"name": tool_name}
    if tool_name == web_search.name:
        args = f' ツール名\n'
        args += f' {tool_name}\n'
        args += f' 引数\n'
        for k, v in tool_args.items():
            args += f' {k}: {v}\n'

        tool_data["args"] = args
    elif tool_name == write_file.name:
        args = f' ツール名\n'
        args += f' {tool_name}\n'
        args += f' 保存ファイル名\n'
        args += f' {tool_args["file_path"]}\n'
        tool_data["html"] = tool_args["text"]
    tool_data["args"] = args

    feedback = interrupt(tool_data)

    if feedback == "APPROVE": # ユーザーがツールの使用を承認
        return tool_call
    
    return ToolMessage(
        content="ツールの利用が拒否されたので、処理を終了してください。",
        name=tool_name,
        tool_call_id=tool_call["id"]
    )

# チェックポインターの設定
checkpointer = MemorySaver()
@entrypoint(checkpointer)
def agent(messages):
    # LLMを呼び出し
    llm_response = invoke_llm(messages).result()

    # ツール呼び出しがある限り繰り返す
    while True:
        if not llm_response.tool_calls:
            break

        approve_tools = []
        tool_results = []

        # 各ツール呼び出しに対してユーザーの承認を求める
        for tool_call in llm_response.tool_calls:
            feedback = ask_human(tool_call)
            if isinstance(feedback, ToolMessage):
                tool_results.append(feedback)
            else:
                approve_tools.append(feedback)

        # 承認されたツールを実行
        tool_futures = []
        for tool_call in approve_tools:
            future = use_tool(tool_call)
            tool_futures.append(future)

        # Futureが完了するのを待って結果だけを集める
        tool_use_results = []
        for future in tool_futures:
            result = future.result()
            tool_use_results.append(result)

        # メッセージリストに追加
        messages = add_messages(
            messages, 
            [llm_response, *tool_use_results, *tool_results]
            )
        
        # 再度LLMを呼び出し
        llm_response = invoke_llm(messages).result()

    return llm_response
