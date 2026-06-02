"""
state.py — 定義 LangGraph StateGraph 的狀態結構

本版本由線性 workflow 升級為 Agentic 架構：
  - 中央 brain 節點讀取目前進度，決定下一步要做什麼
  - 其他節點執行完後一律回到 brain，由 brain 再次決策
  - next_action / brain_reasoning 暴露 agent 的思考軌跡
  - step_count 作為安全網，避免無限迴圈

對話歷史交給 LangGraph 內建的 MessagesState 自動管理。
"""

from langgraph.graph import MessagesState


class TravelState(MessagesState):
    # messages 由 MessagesState 內建，自動以 add_messages reducer 累積
    preferences: str         # Retrieve 動作：RAG 檢索到的偏好資料
    external_info: str       # Search 動作：外部工具蒐集的即時資訊
    draft_itinerary: str     # Generate 動作：行程草案
    reflection: str          # Reflect 動作：可行性評估意見（第一行 PASS/REVISE/RESEARCH）
    revision_count: int      # Generate 動作：已產出第幾版草案（供 brain 判斷是否收斂）
    final_response: str      # Respond 動作：最終回覆內容

    # Brain 控制欄位
    next_action: str         # Brain 決定的下一個動作名稱
    brain_reasoning: str     # Brain 選這個動作的理由（可被讀取顯示）
    step_count: int          # 已執行的 brain 決策次數（迴圈上限保護）
