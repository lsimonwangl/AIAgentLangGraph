"""
Travel Agent - StateGraph 組裝
=============================
graph/builder.py 負責把四個節點組成 LangGraph 主流程，並設定審查迴圈的條件邊。

執行流程：
    0. 載入套件
    1. 定義條件邊 route：審查通過或達修正上限就結束，否則回 planner 修訂
    2. 註冊四個節點並連接固定邊
    3. 以 MemorySaver 編譯（短期記憶，依 thread_id 保留多輪對話，重啟即失）

此模組提供 build_graph() 函式供 main.py 呼叫。
"""

# 載入套件
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from .state import TravelState


def route(state: TravelState):
    """條件邊：verdict 為 pass 或修正次數達上限就結束，否則帶著 review 回 planner 重做。

    刻意保持二元路由：不拉 reviewer → executor 的邊，
    修改方式交給 planner 讀 review 後自行決定。
    """
    review = state.get("review")
    # 教學範例最多審查兩次，避免流程因反覆修訂變得難以理解
    if (review is not None and review["verdict"] == "pass") or state.get("review_count", 0) >= 2:
        return END
    return "planner"


def build_graph(retrieve_preferences, planner, executor, reviewer):
    """組裝並編譯 StateGraph，回傳可執行的 graph 給 main.py 使用。"""
    builder = StateGraph(TravelState)

    # 註冊四個節點
    builder.add_node("retrieve_preferences", retrieve_preferences)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reviewer", reviewer)

    # 連接主幹固定邊與審查迴圈的條件邊
    builder.add_edge(START, "retrieve_preferences")
    builder.add_edge("retrieve_preferences", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reviewer")
    builder.add_conditional_edges("reviewer", route, ["planner", END])

    # 以 MemorySaver 編譯：短期記憶，依 thread_id 保留多輪對話歷史（重啟即失）
    app = builder.compile(checkpointer=MemorySaver())
    print("✅ StateGraph 建構完成")
    return app
