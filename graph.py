"""
Travel Agent - StateGraph 組裝
=============================
graph.py 負責把四個節點組成 LangGraph 主流程，並設定審核迴圈的條件邊。

程式流程：
  1. 定義 route 條件邊，依審核結果決定結束或退回 Planner。
  2. 註冊 Retrieve Preferences、Planner、Executor 與 Reflect 四個節點。
  3. 連接固定邊與 Reflect 後方的條件邊，形成自我修正迴圈。
  4. 使用 MemorySaver 編譯可執行的 StateGraph。
"""

# ── 載入套件 ──────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from openai import APIError

from state import TravelState


# ── 定義 Reflect 後方的條件邊 ─────────────────────────────
def route(state: TravelState):
    """條件邊：verdict 為 pass 或審核輪次達上限就結束，否則帶著 critique 回 planner 重做。

    刻意保持二元路由：不拉 reflect → executor 的邊，
    「該補資料還是該重規劃」的診斷交給 planner 讀 critique 後自行決定。
    """
    # 從 State 取出 Reflect 最新產生的結構化審核結果。
    critique = state.get("critique")
    # verdict 為 pass 代表結果已通過；revisions 達 3 代表已到安全上限。
    # 任一條件成立都回傳 END，避免教學範例產生過長或無限的修訂流程。
    if (critique is not None and critique["verdict"] == "pass") or state.get("revisions", 0) >= 3:
        return END
    # 尚有需要修改的 issues 時，退回 Planner 重新產生修訂計畫。
    return "planner"


# ── 註冊節點並組裝 StateGraph ─────────────────────────────
def build_graph(retrieve_preferences, planner, executor, reflect):
    """組裝並編譯 StateGraph，回傳可執行的 graph 給 main.py 使用。"""
    # 指定 TravelState 為所有節點共用的狀態資料結構。
    builder = StateGraph(TravelState)

    # 將偏好檢索函式註冊成 Retrieve Preferences 節點。
    builder.add_node("retrieve_preferences", retrieve_preferences)
    # 將任務拆解函式註冊成 Planner 節點。
    builder.add_node("planner", planner)
    # 將工具執行函式註冊成 Executor 節點。
    builder.add_node(
        "executor",
        executor,
        # RetryPolicy 是 LangGraph 的節點重試設定；Executor 發生指定錯誤時會重新執行此節點。
        retry_policy=RetryPolicy(
            # max_attempts 是每次進入 Executor 節點後的最大嘗試次數，包含第一次執行與兩次重試。
            max_attempts=3,
            # initial_interval 是第一次重試前的等待秒數，避免立即再次呼叫過載的 API。
            initial_interval=5.0,
            # backoff_factor 是等待時間倍率，因此第二次重試前會等待 5 × 2 = 10 秒。
            backoff_factor=2.0,
            # max_interval 是單次等待秒數上限，避免後續等待時間持續增加。
            max_interval=15.0,
            # jitter 決定是否加入隨機等待時間；教學範例關閉後較容易觀察固定間隔。
            jitter=False,
            # retry_on 指定要重試的錯誤類型，只重試 OpenAI 相容端點的 API 錯誤。
            retry_on=APIError,
        ),
    )
    # 將結果審核函式註冊成 Reflect 節點。
    builder.add_node("reflect", reflect)

    # START 後先檢索使用者過往旅遊偏好。
    builder.add_edge(START, "retrieve_preferences")
    # 偏好資料準備完成後交給 Planner 規劃。
    builder.add_edge("retrieve_preferences", "planner")
    # Planner 產生計畫後交給 Executor 呼叫工具並產生草案。
    builder.add_edge("planner", "executor")
    # Executor 完成草案後交給 Reflect 審核結果。
    builder.add_edge("executor", "reflect")
    # Reflect 後使用 route 動態選擇退回 Planner 或結束流程。
    builder.add_conditional_edges("reflect", route, ["planner", END])

    # MemorySaver 依 thread_id 保存執行狀態，使同一個終端 session 能進行多輪對話。
    app = builder.compile(checkpointer=MemorySaver())
    # 在終端顯示建構完成，方便使用者確認程式已進入互動階段。
    print("✅ StateGraph 建構完成")
    # 回傳已編譯的 graph，供 chat.py 呼叫 astream() 執行。
    return app
