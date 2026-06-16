"""
Travel Agent - StateGraph 組裝
=============================
graph.py 負責把四個節點組成 LangGraph 主流程，並設定審核迴圈的條件邊。

主幹：
    START → profile → planner → executor → reflect → (條件邊) → planner | END

三大 Agentic 能力由結構強制執行，而非靠 prompt 引導：
    Planning    = planner 產出可修改的 plan
    Reflection  = reflect 多面向檢查 + 條件邊決定是否回 planner
    Proactivity = executor 自主呼叫工具蒐集資訊、主動補足與建議
profile 是確定性前置步驟（偏好每次必用，不交給 agent 決定查不查）。

執行流程：
    0. 載入套件
    1. 定義條件邊 route：審核通過或達修正上限就結束，否則回 planner 修訂
    2. 註冊四個節點並連接固定邊
    3. 以 MemorySaver 編譯（短期記憶，依 thread_id 保留多輪對話，重啟即失）

此模組提供 build_graph() 函式供 main.py 呼叫。
"""

# 載入套件
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import TravelState

# 修正次數上限：實測修正過程會收斂（issues 逐輪減少），設 4 避免極端情況下無窮迴圈
MAX_REVISIONS = 4


def route(state: TravelState):
    """條件邊：verdict 為 pass 或修正次數達上限就結束，否則帶著 critique 回 planner 重做。

    刻意保持二元路由：不拉 reflect → executor 的邊，
    「該補資料還是該重規劃」的診斷交給 planner 讀 critique 後自行決定。
    """
    critique = state.get("critique")
    if (critique is not None and critique["verdict"] == "pass") or state.get("revisions", 0) >= MAX_REVISIONS:
        return END
    return "planner"


def build_graph(profile, planner, executor, reflect):
    """組裝並編譯 StateGraph，回傳可執行的 graph 給 main.py 使用。"""
    builder = StateGraph(TravelState)

    # 註冊四個節點
    builder.add_node("profile", profile)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reflect", reflect)

    # 連接主幹固定邊與審核迴圈的條件邊
    builder.add_edge(START, "profile")
    builder.add_edge("profile", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflect")
    builder.add_conditional_edges("reflect", route, ["planner", END])

    # 以 MemorySaver 編譯：短期記憶，依 thread_id 保留多輪對話歷史（重啟即失）
    app = builder.compile(checkpointer=MemorySaver())
    print("✅ StateGraph 建構完成")
    return app
