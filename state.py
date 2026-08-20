"""
Travel Agent - State 定義
========================
state.py 負責定義整張 StateGraph 共享的 TravelState。

程式流程：
  1. 繼承 MessagesState，取得各節點共用的 messages 對話歷史。
  2. 加入 preferences、plan、critique 與 revisions 欄位。
  3. 將 TravelState 提供給各節點與 StateGraph 共用。
"""

# ── 載入套件 ──────────────────────────────────────────────
from langgraph.graph import MessagesState


# ── 定義 StateGraph 共用狀態 ──────────────────────────────
class TravelState(MessagesState):
    """整張 graph 共享的狀態。

    繼承 MessagesState 取得內建的 messages 欄位（多輪對話歷史），
    配合 checkpointer + thread_id 天然支援多 session 隔離。
    """

    # Retrieve Preferences 節點檢索到的偏好原文，供 Planner、Executor 與 Reflect 共用。
    preferences: str
    # Planner 產生的有序步驟清單；Reflect 發現問題時會由 Planner 重新產生。
    plan: list[str]
    # Reflect 的結構化審核結果；轉成 dict 後較適合交給 checkpointer 儲存。
    critique: dict | None
    # 已執行的審核輪次；達到上限時結束迴圈，避免模型無限修訂。
    revisions: int
