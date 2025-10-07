# Langgraphのチェックポイント、Human-in-the-Loop、ストリーミングを活用したエージェント
# AIエージェントはユーザーからのフィードバックを反映して、次の行動を決定します。

from dataclasses import dataclass
from typing import Protocol
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


# LLM呼び出しタスク用の関数
@task
def _invoke_llm_task(llm, messages: list[BaseMessage], system_prompt: str) -> AIMessage:
    """LLMを呼び出すタスク関数

    Args:
        llm: 呼び出すLLMインスタンス
        messages: 会話履歴メッセージ
        system_prompt: システムプロンプト

    Returns:
        AIMessage: LLMの応答
    """
    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    return response


# ツール実行タスク用の関数
@task
def _execute_tool_task(tool, tool_call: ToolCall) -> ToolMessage:
    """ツールを実行するタスク関数

    Args:
        tool: 実行するツール
        tool_call: ツール呼び出し情報

    Returns:
        ToolMessage: ツール実行結果
    """
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])


@dataclass(frozen=True)
class ToolApprovalRequest:
    """ツール実行承認リクエストを表す値オブジェクト

    目的: ツール実行に必要な承認情報をカプセル化し、
         安全な状態でデータを保持する

    Attributes:
        tool_name: ツール名
        display_args: ユーザーに表示する引数情報
        html_content: HTMLプレビュー用コンテンツ(オプショナル)
    """
    tool_name: str
    display_args: str
    html_content: str | None = None

    def to_interrupt_data(self) -> dict:
        """interrupt関数に渡すデータ形式に変換

        Returns:
            dict: interrupt関数用のデータ辞書
        """
        data = {
            "name": self.tool_name,
            "args": self.display_args
        }
        if self.html_content:
            data["html"] = self.html_content
        return data


class ToolApprovalRequestFactory:
    """ツール承認リクエストを生成するファクトリ

    目的: ツール種類に応じた承認リクエスト生成ロジックをカプセル化
    """

    def __init__(self, web_search_tool_name: str, file_write_tool_name: str):
        """
        Args:
            web_search_tool_name: Web検索ツールの名前
            file_write_tool_name: ファイル書き込みツールの名前
        """
        self._web_search_tool_name = web_search_tool_name
        self._file_write_tool_name = file_write_tool_name

    def create_from_tool_call(self, tool_call: ToolCall) -> ToolApprovalRequest:
        """ToolCallから承認リクエストを生成

        Args:
            tool_call: LLMからのツール呼び出し情報

        Returns:
            ToolApprovalRequest: 生成された承認リクエスト
        """
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == self._web_search_tool_name:
            return self._create_web_search_request(tool_name, tool_args)
        elif tool_name == self._file_write_tool_name:
            return self._create_file_write_request(tool_name, tool_args)
        else:
            return self._create_generic_request(tool_name, tool_args)

    def _create_web_search_request(self, tool_name: str, args: dict) -> ToolApprovalRequest:
        """Web検索用の承認リクエストを生成"""
        display_args = f' ツール名\n {tool_name}\n 引数\n'
        for key, value in args.items():
            display_args += f' {key}: {value}\n'
        return ToolApprovalRequest(tool_name=tool_name, display_args=display_args)

    def _create_file_write_request(self, tool_name: str, args: dict) -> ToolApprovalRequest:
        """ファイル書き込み用の承認リクエストを生成"""
        display_args = f' ツール名\n {tool_name}\n 保存ファイル名\n {args["file_path"]}\n'
        return ToolApprovalRequest(
            tool_name=tool_name,
            display_args=display_args,
            html_content=args["text"]
        )

    def _create_generic_request(self, tool_name: str, args: dict) -> ToolApprovalRequest:
        """汎用的な承認リクエストを生成"""
        display_args = f' ツール名\n {tool_name}\n 引数\n {str(args)}\n'
        return ToolApprovalRequest(tool_name=tool_name, display_args=display_args)


class ToolExecutionApprover:
    """ツール実行承認を管理する責務を持つクラス

    目的: Human-in-the-Loopによる承認/拒否の判断をカプセル化
    """

    def __init__(self, request_factory: ToolApprovalRequestFactory):
        """
        Args:
            request_factory: 承認リクエスト生成用ファクトリ
        """
        self._request_factory = request_factory

    def request_approval(self, tool_call: ToolCall) -> ToolCall | ToolMessage:
        """ツール実行の承認をユーザーに要求

        Args:
            tool_call: 承認を求めるツール呼び出し

        Returns:
            ToolCall: 承認された場合は元のツール呼び出し
            ToolMessage: 拒否された場合は拒否メッセージ
        """
        approval_request = self._request_factory.create_from_tool_call(tool_call)
        feedback = interrupt(approval_request.to_interrupt_data())

        if feedback == "APPROVE":
            return tool_call

        return ToolMessage(
            content="ツールの利用が拒否されたので、処理を終了してください。",
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )


class ToolRegistry:
    """ツールの登録と取得を管理する責務を持つクラス

    目的: ツールの初期化とアクセスをカプセル化
    """

    def __init__(self, working_directory: str):
        """
        Args:
            working_directory: ファイル操作の作業ディレクトリ
        """
        self._working_directory = working_directory
        self._tools = self._initialize_tools()
        self._tools_by_name = {tool.name: tool for tool in self._tools}

    def _initialize_tools(self) -> list:
        """ツールの初期化

        Returns:
            list: 初期化されたツールのリスト
        """
        web_search = TavilySearch(max_results=2, topic="general")

        file_toolkit = FileManagementToolkit(
            root_dir=str(self._working_directory),
            selected_tools=["write_file"],
        )
        write_file = file_toolkit.get_tools()[0]

        return [web_search, write_file]

    def get_all_tools(self) -> list:
        """全ツールを取得

        Returns:
            list: 登録されている全ツール
        """
        return self._tools

    def get_tool_by_name(self, name: str):
        """名前でツールを取得

        Args:
            name: ツール名

        Returns:
            取得したツール

        Raises:
            KeyError: 指定された名前のツールが存在しない場合
        """
        if name not in self._tools_by_name:
            raise KeyError(f"ツール '{name}' が見つかりません")
        return self._tools_by_name[name]

    @property
    def web_search_tool_name(self) -> str:
        """Web検索ツールの名前を取得"""
        return self._tools[0].name

    @property
    def file_write_tool_name(self) -> str:
        """ファイル書き込みツールの名前を取得"""
        return self._tools[1].name


class AgentLLMInvoker:
    """エージェントのLLM呼び出しを管理する責務を持つクラス

    目的: LLMの初期化と呼び出しロジックをカプセル化
    """

    SYSTEM_PROMPT = """
あなたの責務はユーザーからのリクエストを調査し、調査結果をファイルに出力することです。
- ユーザーのリクエスト調査にWeb検索が必要であれば、Web検索ツールを使用してください。
- 必要な情報が集まったと判断したら検索は終了して下さい。
- 検索は最大2回までにして下さい。
- ファイル形式はHTML形式に変換して保存して下さい。
  * Web検索が拒否された場合、Web検索を中止して下さい。
  * レポート保存が拒否された場合、レポート保存を中止し、内容をユーザーに伝えて下さい。
"""

    def __init__(self, tools: list):
        """
        Args:
            tools: LLMにバインドするツールのリスト
        """
        self._llm = self._initialize_llm(tools)

    def _initialize_llm(self, tools: list):
        """LLMを初期化

        Args:
            tools: バインドするツール

        Returns:
            初期化されたLLM
        """
        cfg = Config(read_timeout=300)
        return init_chat_model(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_provider="bedrock_converse",
            config=cfg,
        ).bind_tools(tools)

    def invoke(self, messages: list[BaseMessage]):
        """LLMを呼び出す(タスクとして実行)

        Args:
            messages: 会話履歴メッセージ

        Returns:
            Future: LLM呼び出しの非同期結果
        """
        return _invoke_llm_task(self._llm, messages, self.SYSTEM_PROMPT)


class ToolExecutor:
    """ツール実行を管理する責務を持つクラス

    目的: ツールの実行ロジックをカプセル化
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Args:
            tool_registry: ツールレジストリ
        """
        self._tool_registry = tool_registry

    def execute(self, tool_call: ToolCall):
        """ツールを実行(タスクとして実行)

        Args:
            tool_call: 実行するツール呼び出し

        Returns:
            Future: ツール実行の非同期結果
        """
        tool = self._tool_registry.get_tool_by_name(tool_call["name"])
        return _execute_tool_task(tool, tool_call)


class ResearchAgent:
    """調査エージェントのオーケストレーションを管理するクラス

    目的: エージェント全体の実行フローを統括し、
         各コンポーネント間の協調動作を管理
    """

    def __init__(
        self,
        llm_invoker: AgentLLMInvoker,
        tool_executor: ToolExecutor,
        approval_manager: ToolExecutionApprover
    ):
        """
        Args:
            llm_invoker: LLM呼び出し管理
            tool_executor: ツール実行管理
            approval_manager: ツール承認管理
        """
        self._llm_invoker = llm_invoker
        self._tool_executor = tool_executor
        self._approval_manager = approval_manager

    def run(self, messages: list[BaseMessage]) -> AIMessage:
        """エージェントを実行

        Args:
            messages: 初期メッセージリスト

        Returns:
            AIMessage: 最終的なLLM応答
        """
        llm_response = self._llm_invoker.invoke(messages).result()

        while llm_response.tool_calls:
            approved_tools, rejection_messages = self._process_tool_approvals(
                llm_response.tool_calls
            )

            tool_results = self._execute_approved_tools(approved_tools)

            messages = add_messages(
                messages,
                [llm_response, *tool_results, *rejection_messages]
            )

            llm_response = self._llm_invoker.invoke(messages).result()

        return llm_response

    def _process_tool_approvals(
        self,
        tool_calls: list[ToolCall]
    ) -> tuple[list[ToolCall], list[ToolMessage]]:
        """ツール承認処理を実行

        Args:
            tool_calls: 承認を求めるツール呼び出しリスト

        Returns:
            tuple: (承認されたツールリスト, 拒否メッセージリスト)
        """
        approved_tools = []
        rejection_messages = []

        for tool_call in tool_calls:
            feedback = self._approval_manager.request_approval(tool_call)
            if isinstance(feedback, ToolMessage):
                rejection_messages.append(feedback)
            else:
                approved_tools.append(feedback)

        return approved_tools, rejection_messages

    def _execute_approved_tools(self, approved_tools: list[ToolCall]) -> list[ToolMessage]:
        """承認されたツールを並列実行

        Args:
            approved_tools: 実行するツール呼び出しリスト

        Returns:
            list[ToolMessage]: ツール実行結果リスト
        """
        tool_futures = [
            self._tool_executor.execute(tool_call)
            for tool_call in approved_tools
        ]

        return [future.result() for future in tool_futures]


# エージェントの初期化とエントリーポイント設定
_tool_registry = ToolRegistry(working_directory="report")
_llm_invoker = AgentLLMInvoker(tools=_tool_registry.get_all_tools())
_tool_executor = ToolExecutor(tool_registry=_tool_registry)
_request_factory = ToolApprovalRequestFactory(
    web_search_tool_name=_tool_registry.web_search_tool_name,
    file_write_tool_name=_tool_registry.file_write_tool_name
)
_approval_manager = ToolExecutionApprover(request_factory=_request_factory)
_research_agent = ResearchAgent(
    llm_invoker=_llm_invoker,
    tool_executor=_tool_executor,
    approval_manager=_approval_manager
)

checkpointer = MemorySaver()

@entrypoint(checkpointer)
def agent(messages: list[BaseMessage]) -> AIMessage:
    """エージェントのエントリーポイント

    Args:
        messages: 処理する初期メッセージ

    Returns:
        AIMessage: エージェントの最終応答
    """
    return _research_agent.run(messages)
