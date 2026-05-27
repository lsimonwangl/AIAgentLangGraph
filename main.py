"""
main.py — 主程式入口

串接所有模組，依序完成：
  1. 建立 RAG 檢索器（rag.py）
  2. 初始化 LLM（NVIDIA API）
  3. 載入外部工具（tools.py）
  4. 建立節點函式（nodes.py）
  5. 建構 StateGraph（graph.py）
  6. 啟動對話迴圈（chat.py）
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import build_retriever
from tools import get_all_tools
from nodes import create_nodes
from graph import build_graph
from chat import chat_loop

load_dotenv()


async def main():
    # ── 1. 建立 RAG 檢索器 ──
    print("正在建立 RAG 檢索器...")
    retriever = build_retriever()

    # ── 2. 初始化 LLM ──
    print("正在初始化 LLM...")
    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.7,
        stream_chunk_timeout=300,  # NVIDIA 共享 API 偶爾排隊較久，放寬至 5 分鐘
    )

    # ── 3. 載入外部工具（MCP tools） ──
    print("正在載入外部工具...")
    all_tools = await get_all_tools()

    # ── 4. 建立節點函式 ──
    print("正在建立節點函式...")
    nodes = create_nodes(llm, retriever, all_tools)

    # ── 5. 建構 StateGraph ──
    print("正在建構 StateGraph...")
    app = build_graph(nodes)

    # ── 6. 啟動對話迴圈 ──
    await chat_loop(app)


if __name__ == "__main__":
    asyncio.run(main())
