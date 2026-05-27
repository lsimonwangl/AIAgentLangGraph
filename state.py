"""
state.py — 定義 LangGraph StateGraph 的狀態結構

StateGraph 中所有節點共享同一份 State，
每個節點讀取所需欄位、寫入自己負責的欄位，
State 隨著圖的執行逐步填充完整。
"""

from typing import TypedDict


class TravelState(TypedDict):
    user_query: str          # 使用者原始問題
    chat_history: str        # 累積的多輪對話紀錄
    plan: str                # Plan 節點：任務拆解結果
    preferences: str         # Retrieve 節點：RAG 檢索到的偏好資料
    external_info: str       # Search 節點：外部工具蒐集的即時資訊
    draft_itinerary: str     # Generate 節點：行程草案（含預算）
    reflection: str          # Reflect 節點：可行性評估意見
    is_feasible: bool        # Reflect 節點：是否通過評估
    revision_count: int      # Reflect 節點：已評估次數（用於限制迴圈）
    final_response: str      # Respond 節點：最終回覆內容
