"""
Travel Agent - Retrieve Preferences 節點（偏好前置檢索）
=====================================================
retrieve_preferences.py 負責在 planner 之前檢索使用者過往旅遊偏好，存入共享 state。

執行流程：
    0. 從對話歷史取出最新一則使用者需求
    1. 透過 template.build_preference_query 將需求改寫成偏好搜尋文字
    2. 用 retriever 從 Milvus 檢索相關偏好片段
    3. 原文片段直接存入 state["preferences"]，供 planner/executor/reflect 共用
       （不經 LLM 摘要：原文天然就是行程說明要引用的依據，摘要反而有失真風險）

此模組提供 create_retrieve_preferences() 函式供 main.py 呼叫。
"""

# 載入套件
from langchain_core.messages import HumanMessage

from state import TravelState
from template import build_preference_query


def create_retrieve_preferences(retriever):
    """建立 retrieve_preferences 節點，回傳可註冊進 StateGraph 的 async 函式。"""

    async def retrieve_preferences(state: TravelState) -> dict:
        # 取出最新一則使用者訊息，作為改寫檢索詞的依據
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        # 將需求改寫成偏好搜尋文字後檢索，每輪新需求都重撈一次（純向量查詢，成本可忽略）
        docs = retriever.invoke(build_preference_query(user_query))
        return {"preferences": "\n\n".join(doc.page_content for doc in docs)}

    return retrieve_preferences
