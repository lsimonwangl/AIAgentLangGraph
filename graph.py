"""
graph.py — Agentic StateGraph 建構模組

由「線性 workflow + Reflect 條件邊」升級為「中央 brain 路由」：

  ┌────────────────────────────────────┐
  ↓                                    │
brain ──conditional──→ retrieve ───────┤
                  ├─→ search ──────────┤
                  ├─→ generate ─→ reflect ─┘
                  └─→ respond → END

brain 決定下一步要 retrieve / search / generate / finish。
其中 generate 完成後「固定接 reflect」（架構強制：每次生成都一定評估，
brain 無法跳過），reflect 完成後回到 brain 再決策。retrieve / search 完成後也回 brain。
沒有寫死的整體順序——agent 可以反思多次、可以重查資訊、可以提早 finish。
step_count 設上限，作為避免無限迴圈的安全網。
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import TravelState

MAX_STEPS = 20  # brain 決策次數上限，超過強制 finish


def route_from_brain(state: TravelState) -> str:
    """根據 brain 的決策路由到下一個動作節點。"""
    # 安全網：超過上限直接 respond，避免無限迴圈
    if state.get("step_count", 0) >= MAX_STEPS:
        return "respond"

    action = state.get("next_action", "finish")
    if action == "finish":
        return "respond"
    return action


def build_graph(nodes: dict):
    """
    建構並編譯 Agentic StateGraph。

    Parameters:
        nodes: create_nodes() 回傳的節點函式對應表（需包含 brain）
    Returns:
        CompiledGraph: 編譯後的可執行圖
    """
    graph = StateGraph(TravelState)

    # ── 註冊節點 ──
    graph.add_node("brain", nodes["brain"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("search", nodes["search"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("reflect", nodes["reflect"])
    graph.add_node("respond", nodes["respond"])

    # ── 設定進入點 ──
    graph.set_entry_point("brain")

    # ── Brain 的條件邊：依 next_action 路由（brain 不選 reflect，reflect 由 generate 自動觸發）──
    graph.add_conditional_edges(
        "brain",
        route_from_brain,
        {
            "retrieve": "retrieve",
            "search": "search",
            "generate": "generate",
            "respond": "respond",
        },
    )

    # ── generate 完成後固定接 reflect：架構強制每次生成都評估，brain 無法跳過 ──
    graph.add_edge("generate", "reflect")

    # ── retrieve / search / reflect 完成後回到 brain 再決策 ──
    for action_node in ("retrieve", "search", "reflect"):
        graph.add_edge(action_node, "brain")

    # ── Respond 是終點 ──
    graph.add_edge("respond", END)

    # ── 編譯（使用 MemorySaver 支援多輪對話） ──
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    print("Agentic StateGraph 建構完成（brain 路由模式）")
    return app
