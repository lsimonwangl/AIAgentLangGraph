"""
Travel Agent - 主程式入口
=======================
main.py 負責把偏好檢索、四個節點與 StateGraph 串成完整旅遊規劃流程。

程式流程：
  1. 載入 Executor 可以使用的 MCP 外部工具。
  2. 建立搜尋過往旅遊紀錄的 retriever。
  3. 初始化 LLM 與四個 LangGraph 節點。
  4. 組裝具有審核迴圈與短期記憶的 StateGraph。
  5. 啟動終端機介面，接收使用者的多輪旅遊需求。

執行方式：
    python main.py
"""

# ── 載入套件與環境變數 ──────────────────────────────────
from dotenv import load_dotenv

# 讀取專案根目錄的 .env，提供模型與工具所需的 API 設定。
load_dotenv()

import asyncio
import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from chat import run_chat
from executor import create_executor
from graph import build_graph
from planner import create_planner
from rag import build_retriever
from reflect import create_reflect
from retrieve_preferences import create_retrieve_preferences
from tools import load_mcp_tools


# ── 初始化元件並啟動 StateGraph ───────────────────────────
async def main():
    # 啟動三個 MCP Server，取得 Tavily 搜尋、Open-Meteo 天氣與 Frankfurter 匯率工具。
    # mcp_client 保留在區域變數，讓終端機對話期間的工具連線持續有效。
    mcp_client, tools = await load_mcp_tools()

    # 讀取 data/ 旅遊紀錄並建立 Milvus Retriever，供偏好檢索節點使用。
    retriever = build_retriever()

    # 使用 ChatOpenAI 連接 NVIDIA 的 OpenAI 相容 API。
    llm = ChatOpenAI(
        # NVIDIA NIM 提供的 OpenAI 相容端點。
        base_url="https://integrate.api.nvidia.com/v1",
        # API 金鑰從 .env 的 NVIDIA_API_KEY 讀取，避免直接寫在程式碼中。
        api_key=os.getenv("NVIDIA_API_KEY"),
        # 模型名稱從 .env 的 CHAT_MODEL 讀取，方便不改程式就能切換模型。
        model=os.getenv("CHAT_MODEL"),
        # 放寬串流等待至 5 分鐘，容忍共享 API 排隊或較長回答。
        stream_chunk_timeout=300,
        # 最多自動重試兩次，避免遇到 429 時產生大量連續請求。
        max_retries=2,
        # 主動限制約 9 RPM，且不允許短時間爆發，預留 API 速率上限空間。
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.15, max_bucket_size=1),
    )

    # 將 Retriever 包裝成每輪都會執行的偏好前置檢索節點。
    retrieve_preferences = create_retrieve_preferences(retriever)
    # 將同一個 LLM 包裝成負責任務拆解與修訂的 Planner 節點。
    planner = create_planner(llm)
    # 將 LLM 與 MCP 工具組合成可自主執行計畫的 Executor 節點。
    executor = create_executor(llm, tools)
    # 將同一個 LLM 包裝成只負責檢查最終行程草案的 Reflect 節點。
    reflect = create_reflect(llm)

    # 註冊四個節點並連接固定邊與條件邊，組成可循環修訂的 StateGraph。
    graph = build_graph(retrieve_preferences, planner, executor, reflect)

    # 將已編譯的 graph 交給終端機介面，開始接收多輪旅遊需求。
    await run_chat(graph)


# ── 執行主程式 ────────────────────────────────────────────
if __name__ == "__main__":
    # main() 是 async 函式，使用 asyncio.run() 建立事件迴圈並執行。
    asyncio.run(main())
