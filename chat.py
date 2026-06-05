# ============================================================
# chat.py — 對話迴圈、串流顯示與 interrupt 暫停/resume
# ============================================================
# 逐節點顯示 StateGraph 執行進度（Planner → Executor → Reflect），
# 並保留通用的 interrupt 處理：節點若觸發 interrupt 則暫停 → 終端反問 → Command(resume) 續跑
# （目前無節點觸發，為保留 human-in-the-loop 能力的基礎設施）。
# 串流用 stream_mode=["updates","messages"] + subgraphs=True：
#   - messages：即時 token（planner 計畫、executor 行程草案）
#   - updates ：節點產出的 state 欄位（plan / critique），與子圖工具呼叫
# ============================================================

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from prompt import read_query


# ===================== 顯示小工具 =====================
def preview_text(text: str, max_chars: int = 400) -> str:
    text = str(text).strip()
    if not text:
        return "（空）"
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}...（共 {len(text)} 字元）"


def print_header(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# 串流中目前的標題（避免同一節點重複印 header）
_HEADERS = {
    "planner": "Planner — 規劃 / 修訂計畫",
    "executor": "Executor — 執行與生成（ReAct）",
}


# ===================== 子圖工具呼叫顯示 =====================
# executor 是 create_agent 子圖，其工具呼叫/結果走 subgraph 的 updates。
def _print_tool_activity(node_output: dict):
    for value in node_output.values():
        for msg in value.get("messages", []) if isinstance(value, dict) else []:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[Tool Call] {tc['name']}")
                    print(f"  參數：{preview_text(tc['args'], 120)}")
            elif isinstance(msg, ToolMessage):
                print(f"  結果：{preview_text(msg.content, 150)}", flush=True)


# ===================== 串流事件處理 =====================
def _handle_event(ns, mode, data, st: dict):
    # ── 子圖（executor 內部）事件 ──
    if ns:
        if mode == "messages":
            chunk, _meta = data
            # 只串流 LLM 的推理/行程文字；工具回傳的 ToolMessage 內容不在此原文倒出，
            # 改由 _print_tool_activity 印截斷後的「結果：」即可。
            if chunk.content and not isinstance(chunk, ToolMessage):
                if st["header"] != "executor":
                    print_header(_HEADERS["executor"])
                    st["header"] = "executor"
                print(chunk.content, end="", flush=True)
        elif mode == "updates":
            _print_tool_activity(data)
        return

    # ── 頂層事件 ──
    if mode == "messages":
        chunk, meta = data
        node = meta.get("langgraph_node")
        if node == "planner" and chunk.content:
            if st["header"] != "planner":
                print_header(_HEADERS["planner"])
                st["header"] = "planner"
            print(chunk.content, end="", flush=True)
        return

    # mode == "updates"
    for node_name, output in data.items():
        if node_name == "__interrupt__":
            continue  # interrupt 由外層 get_state 統一處理
        if node_name == "planner":
            plan = output.get("plan", [])
            print('\n\n輸出欄位：state["plan"]（可修改的計畫物件）')
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step}")
            st["header"] = None
        elif node_name == "executor":
            st["header"] = None  # 行程草案已串流完
        elif node_name == "reflect":
            critique = output.get("critique")
            count = output.get("revisions", 0)
            print_header("Reflect — 多面向品質檢查")
            print('輸出欄位：state["critique"]（structured output）')
            print(f"  verdict：{critique['verdict'] if critique else '?'}")
            print(f"  revisions：{count}")
            issues = critique["issues"] if critique else []
            if issues:
                print("  issues：")
                for issue in issues:
                    print(f"    ・{issue}")
            st["header"] = None


# ===================== 單輪執行（含 interrupt 迴圈） =====================
# 串流狀態：跨事件記住目前 header，避免同一節點重複印標題
_STREAM_STATE = {"header": None}


async def _run_turn(graph, payload, config):
    async for ns, mode, data in graph.astream(
        payload, config, stream_mode=["updates", "messages"], subgraphs=True
    ):
        _handle_event(ns, mode, data, _STREAM_STATE)
    snapshot = await graph.aget_state(config)
    return snapshot.interrupts


# ===================== 對話迴圈 =====================
async def chat_loop(graph):
    config = {"configurable": {"thread_id": "travel-session-1"}}

    print_header("個人化旅遊規劃 Agentic AI（LangGraph）")
    print("  輸入旅遊需求開始規劃，輸入 quit 結束")

    while True:
        user_input = read_query()
        if user_input.lower() == "quit":
            print("\n感謝使用，再見！")
            break
        if not user_input:
            continue

        # 本輪輸入：丟新的 HumanMessage（add_messages 自動 append），
        # 並重置 transient 欄位，避免上一輪殘留干擾。
        payload = {
            "messages": [HumanMessage(content=user_input)],
            "plan": [],
            "critique": None,
            "revisions": 0,
        }

        # interrupt 迴圈：暫停就反問，取得回覆後 Command(resume) 續跑。
        while True:
            _STREAM_STATE["header"] = None
            interrupts = await _run_turn(graph, payload, config)
            if not interrupts:
                break
            answer = input(f"\n[需要補充資訊] {interrupts[0].value}\n> ").strip()
            payload = Command(resume=answer)

        print("\n\n（本輪規劃完成）")
