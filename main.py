"""
Travel Agent - 主程式入口
=======================
main.py 負責把偏好檢索、四個節點與 StateGraph 串成完整旅遊規劃流程。

執行流程：
    0. 載入套件與環境變數
    1. 載入 executor 可以使用的 MCP 外部工具
    2. 建立可以搜尋過往旅遊紀錄的 retriever
    3. 初始化 LLM，建立 retrieve_preferences / planner / executor / reflect 四個節點
    4. 組裝 StateGraph（retrieve_preferences → planner → executor → reflect 審核迴圈）
    5. 啟動終端機互動介面，接收使用者多輪問題

執行方式：
    python main.py
"""

# 載入套件與環境變數工具
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 載入套件
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


async def main():
    # 載入外部工具：Tavily 搜尋景點、open-meteo 查天氣、frankfurter 換匯率
    # mcp_client 保留在區域變數，讓終端機對話期間工具連線持續有效
    mcp_client, tools = await load_mcp_tools()

    # 建立 RAG 檢索器，用來從過往旅遊紀錄找出相關偏好
    retriever = build_retriever()

    # 初始化 LLM，使用 NVIDIA API（OpenAI 相容）
    llm = ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        model=os.getenv("CHAT_MODEL"),  # .env 需設成 NVIDIA 模型
        stream_chunk_timeout=300,  # 放寬回應等待至 5 分鐘，容忍共享 API 排隊
        max_retries=2,  # 限制 retry 次數，避免 429 時連發請求形成 retry storm
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.15, max_bucket_size=1),  # 主動降速約 9 RPM 並關閉爆發，留 RPM 上限安全邊際
    )

    # 建立四個節點：偏好前置檢索、規劃、執行、審核
    retrieve_preferences = create_retrieve_preferences(retriever)
    planner = create_planner(llm)
    executor = create_executor(llm, tools)
    reflect = create_reflect(llm)

    # 組裝 StateGraph，把節點串成「規劃 → 執行 → 審核」的迴圈
    graph = build_graph(retrieve_preferences, planner, executor, reflect)

    # 啟動終端機介面，讓使用者可以一輪一輪輸入需求
    await run_chat(graph)


if __name__ == "__main__":
    asyncio.run(main())
