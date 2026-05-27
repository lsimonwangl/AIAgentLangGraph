"""
graph.py — StateGraph 建構模組

以 LangGraph 的 StateGraph 明確定義節點與條件邊，
將 Agent 的 Plan → Execute → Reflect 流程
從 Prompt 引導升級為架構層級的強制執行。

核心設計：
  - 六個節點依序串接
  - Reflect 節點後加入條件邊（conditional edge）
  - 預算不足 → 自動回到 Generate 重新規劃（Proactivity）
  - 通過評估 → 流向 Respond 回覆使用者
  - 最多評估 2 次，避免無限迴圈
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import TravelState


def should_revise(state: TravelState) -> str:
    """
    條件判斷函式：決定 Reflect 後要重新規劃還是回覆使用者。

    回傳 "revise" → 回到 Generate 節點重新規劃
    回傳 "pass"   → 前往 Respond 節點回覆使用者
    """
    # 已通過可行性評估
    if state.get("is_feasible", False):
        return "pass"

    # 已達最大評估次數，強制通過避免無限迴圈
    if state.get("revision_count", 0) >= 2:
        return "pass"

    # 未通過且未達上限，回到 Generate 重新規劃
    return "revise"


def build_graph(nodes: dict):
    """
    建構並編譯 StateGraph。

    Parameters:
        nodes: create_nodes() 回傳的節點函式對應表
    Returns:
        CompiledGraph: 編譯後的可執行圖
    """
    graph = StateGraph(TravelState)

    # ── 註冊節點 ──
    graph.add_node("plan", nodes["plan"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("search", nodes["search"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("reflect", nodes["reflect"])
    graph.add_node("respond", nodes["respond"])

    # ── 設定進入點 ──
    graph.set_entry_point("plan")

    # ── 設定邊（線性串接） ──
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "reflect")

    # ── 設定條件邊（Reflect 後的分支） ──
    graph.add_conditional_edges(
        "reflect",
        should_revise,
        {
            "revise": "generate",   # 預算不足 → 回到 Generate
            "pass": "respond",      # 通過評估 → 前往 Respond
        },
    )

    # ── 設定結束點 ──
    graph.add_edge("respond", END)

    # ── 編譯（使用 MemorySaver 支援多輪對話） ──
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    print("StateGraph 建構完成")
    return app
