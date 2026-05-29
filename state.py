"""
state.py — 定義 LangGraph StateGraph 的狀態結構

StateGraph 中所有節點共享同一份 State，
每個節點讀取所需欄位、寫入自己負責的欄位，
State 隨著圖的執行逐步填充完整。

對話歷史交給 LangGraph 內建的 MessagesState：
  - messages 欄位由 add_messages reducer 自動累積
  - 不再手動拼接 chat_history 字串
  - 配合 checkpointer + thread_id 天然支援多 session 隔離
"""

from langgraph.graph import MessagesState


class TravelState(MessagesState):
    # messages 欄位由 MessagesState 內建，自動以 add_messages reducer 累積
    plan: str                # Plan 節點：任務拆解結果
    preferences: str         # Retrieve 節點：RAG 檢索到的偏好資料
    external_info: str       # Search 節點：外部工具蒐集的即時資訊
    draft_itinerary: str     # Generate 節點：行程草案（含預算）
    reflection: str          # Reflect 節點：可行性評估意見
    is_feasible: bool        # Reflect 節點：是否通過評估
    revision_count: int      # Reflect 節點：已評估次數（用於限制迴圈）
    final_response: str      # Respond 節點：最終回覆內容
