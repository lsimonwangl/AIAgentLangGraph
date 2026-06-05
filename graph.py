# ============================================================
# graph.py — 組裝 StateGraph
# ============================================================
# 最精簡的 LangGraph 慣用型，三節點主幹：
#   START → planner → executor → reflect → (條件邊) → planner | END
# 三大 Agentic 能力由結構強制執行，而非靠 prompt 引導：
#   Planning   = planner 產出可修改的 plan
#   Reflection = reflect 多面向檢查 + 條件邊決定是否回 planner
#   Proactivity= executor 自主呼叫工具蒐集資訊、主動補足與建議
# checkpointer 用 MemorySaver——保留 interrupt() 暫停/resume 的基礎設施（目前無節點觸發）。
# ============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import TravelState


# ===================== 條件邊 =====================
# verdict == pass 或修正次數達上限 → 結束；否則帶著 critique 回 planner 重做。
# 刻意保持二元路由：不拉 reflect → executor 的邊，「該補資料還是該重規劃」
# 的診斷交給 planner 讀 critique 後自行決定。
# 上限設 4：實測修正過程會收斂（issues 逐輪減少），上限 4 常在快收斂時被切斷。
MAX_REVISIONS = 4


def route(state: TravelState):
    critique = state.get("critique")
    if (critique is not None and critique["verdict"] == "pass") or state.get("revisions", 0) >= MAX_REVISIONS:
        return END
    return "planner"


# ===================== 建構並編譯 =====================
def build_graph(planner, executor, reflect):
    builder = StateGraph(TravelState)

    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reflect", reflect)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflect")
    builder.add_conditional_edges("reflect", route, ["planner", END])

    app = builder.compile(checkpointer=MemorySaver())
    print("StateGraph 建構完成")
    return app
