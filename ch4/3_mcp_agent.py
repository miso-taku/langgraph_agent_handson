import asyncio
import operator
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing import Annotated, Dict, List, Union

from  dotenv import load_dotenv


load_dotenv()

mcp_client = None
tools = None
llm_with_tools = None

async def initialize_llm():
    """MCPクライアントとLLMを初期化する非同期関数"""
    global mcp_client, tools, llm_with_tools

    # MCPクライアントの初期化
    mcp_client = MultiServerMCPClient(
        {
            # ファイルシステムMCPサーバー
            "file-system": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "./"
                ],
                "transport": "stdio",  
            },
            # AWS Knowledge MCPサーバー
            "aws-knowledge-mcp-server": {
                "url": "https://knowledge-mcp.global.api.aws",
                "transport": "streamable_http",
            }
        }
    )
    # MCPサーバーをlangchainツールとして取得
    tools = await mcp_client.get_tools()

    # LLMの初期化
    llm_with_tools = init_chat_model(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_provider="bedrock_converse"
    ).bind_tools(tools)

# ステートの定義
class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add]

system_prompt = """
あなたの責務はAWSドキュメントを検索し、Markdown形式でファイルに出力することです。
- 検索後、Markdown形式に変換してください。
- 検索は最大2回までにしてください。
"""

async def agent(state: AgentState) -> Dict[str, list[AIMessage]]:
    
    response = await llm_with_tools.ainvoke(
        [SystemMessage(system_prompt)] + state.messages
    )

    return {"messages": [response]}

def route_node(state: AgentState) -> Union[str]:
    lastmessage = state.messages[-1]
    if not isinstance(lastmessage, AIMessage):
        raise ValueError("[AIMessage]以外のメッセージ。遷移が不正な可能性があります。")
    if not lastmessage.tool_calls:
        return END # ENDノードへ遷移
    return "tools" # toolsノードへ遷移

async def main():
    # MCPクライアントとLLMの初期化
    await initialize_llm()

    # グラフを構築
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_node)
    builder.add_edge("tools", "agent")

    graph = builder.compile(name="ReACT Agent with MCP Tools")

    question = "AWS Lambdaとは何ですか？"
    response = await graph.ainvoke(
        {
            "messages": [HumanMessage(question)]
        
        }
    )
    print(response)
    return response

asyncio.run(main())