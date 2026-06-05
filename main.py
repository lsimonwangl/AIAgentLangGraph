# ============================================================
# main.py — 主程式入口
# ============================================================
# 依序：建 RAG 檢索器 → 初始化 LLM → 組裝工具 → 建三個節點
#       → 建 StateGraph → 啟動對話迴圈（含 interrupt 暫停/resume）。
# ============================================================

import os
import sys
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 強制終端輸出走 UTF-8：行程內容含「・」「円」等字元，
# Windows 預設 cp950 console 會在 print 時拋 UnicodeEncodeError。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag import build_retriever
from tools import build_tools
from planner import create_planner
from executor import create_executor
from reflect import create_reflect
from graph import build_graph
from chat import chat_loop

load_dotenv()


async def main():
    # ── 1. RAG 檢索器 ──
    print("正在建立 RAG 檢索器...")
    retriever = build_retriever()

    # ── 2. LLM（NVIDIA 端點） ──
    print("正在初始化 LLM...")
    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.7,
        stream_chunk_timeout=300,  # NVIDIA 共享 API 偶爾排隊較久，放寬至 5 分鐘
    )

    # ── 3. 工具（RAG + MCP） ──
    print("正在組裝工具...")
    tools = await build_tools(retriever, llm)

    # ── 4. 三個節點 ──
    print("正在建立節點...")
    planner = create_planner(llm)
    executor = create_executor(llm, tools)
    reflect = create_reflect(llm)

    # ── 5. StateGraph ──
    print("正在建構 StateGraph...")
    app = build_graph(planner, executor, reflect)

    # ── 6. 對話迴圈 ──
    await chat_loop(app)


if __name__ == "__main__":
    asyncio.run(main())
