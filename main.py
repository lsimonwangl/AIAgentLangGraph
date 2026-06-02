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
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 靜音 with_structured_output(BrainDecision) 觸發的無害序列化警告。
# 根源是 LangChain 內部 parsed 欄位型別不符 + Pydantic 2.12 檢查變嚴格（issue #35538），
# 不影響 brain 決策正確性；待 langchain-core 修好後可移除此行。
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

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
