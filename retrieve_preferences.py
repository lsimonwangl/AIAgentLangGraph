"""
Travel Agent - Retrieve Preferences 節點（偏好前置檢索）
=====================================================
retrieve_preferences.py 負責在 planner 之前檢索使用者過往旅遊偏好，存入共享 state。

程式流程：
  1. 從對話歷史取出最新一則使用者需求。
  2. 將需求改寫成適合搜尋過往旅遊偏好的查詢文字。
  3. 使用 retriever 從 Milvus 檢索相關偏好原文。
  4. 將原文存入 state["preferences"]，供後續三個節點共用。
"""

# ── 載入套件 ──────────────────────────────────────────────
from langchain_core.messages import HumanMessage

from state import TravelState
from template import build_preference_query


# ── 建立偏好前置檢索節點 ──────────────────────────────────
def create_retrieve_preferences(retriever):
    """建立 retrieve_preferences 節點，回傳可註冊進 StateGraph 的 async 函式。"""

    # 內部 async 函式就是實際註冊到 StateGraph 的節點。
    async def retrieve_preferences(state: TravelState) -> dict:
        # 對話歷史由新到舊搜尋，第一則 HumanMessage 就是本輪最新需求。
        # 若歷史中沒有使用者訊息，next() 會使用空字串作為預設值，避免拋出例外。
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        # 先將旅遊需求改寫成偏好搜尋文字，再從 Milvus 取回最相關的旅行紀錄片段。
        # 每輪都重新檢索，確保多輪對話的新需求會取得對應的偏好資料。
        docs = retriever.invoke(build_preference_query(user_query))
        # 將所有命中片段以空行串接，直接寫入 State 供後續節點讀取。
        return {"preferences": "\n\n".join(doc.page_content for doc in docs)}

    # 回傳節點函式本身，main.py 之後會把它註冊到 StateGraph。
    return retrieve_preferences
