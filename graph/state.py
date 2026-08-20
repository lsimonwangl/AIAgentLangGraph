"""
Travel Agent - Graph State 定義
==============================
graph/state.py 負責定義整張 StateGraph 共享的 TravelState。

執行流程：
    0. 載入套件
    1. 定義 TravelState：繼承 MessagesState，補上偏好、計畫、審查結果與審查輪數欄位

此模組提供 TravelState 供各節點模組使用。
"""

# 載入套件
from langgraph.graph import MessagesState


class TravelState(MessagesState):
    """整張 graph 共享的狀態。

    繼承 MessagesState 取得內建的 messages 欄位（多輪對話歷史），
    配合 checkpointer + thread_id 天然支援多 session 隔離。
    """

    preferences: str       # retrieve_preferences 節點檢索的偏好原文片段，三個後續節點共用
    plan: list[str]        # planner 產出的可修改計畫（Planning 核心）
    review: dict | None    # reviewer 的審查結果（存 dict 以利 checkpoint 序列化）
    review_count: int      # 審查已執行幾輪，達上限就結束迴圈避免無窮修訂
