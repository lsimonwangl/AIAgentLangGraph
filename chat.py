"""
Travel Agent - 終端機輸出
=======================
chat.py 負責顯示啟動畫面、串流各節點執行進度，並驅動多輪對話迴圈。

執行流程：
    0. 顯示啟動 banner 與範例問題
    1. 透過 prompt.read_query 接收使用者輸入
    2. 呼叫 graph.astream 串流接收節點事件（updates + messages，含子圖）
    3. 依節點類型分別顯示偏好檢索、計畫、工具呼叫、行程草案與審核結果
    4. 使用者結束輸入時印出告別訊息

串流用 stream_mode=["updates","messages"] + subgraphs=True：
    - messages：即時 token（executor 行程草案；planner/reflect 是 structured output，無文字可串）
    - updates ：節點產出的 state 欄位（preferences / plan / critique），與子圖工具呼叫

此模組提供 run_chat() 函式供 main.py 呼叫。
"""

# 載入套件
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from prompt import read_query


def preview_text(text: str, max_chars: int = 400) -> str:
    """截斷過長文字，避免工具回傳洗版終端機。"""
    text = str(text).strip()
    if not text:
        return "（空）"
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}...（共 {len(text)} 字元）"


def print_header(title: str):
    """印出節點區段標題，讓使用者知道 graph 跑到哪個階段。"""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _print_tool_activity(node_output: dict):
    """顯示 executor 子圖的工具呼叫與回傳結果。

    executor 是 create_agent 子圖，其工具呼叫/結果走 subgraph 的 updates 事件。
    """
    for value in node_output.values():
        if not isinstance(value, dict):
            continue
        for msg in value.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # Agent 決定呼叫工具：印出工具名稱與參數，讓使用者知道它正在查什麼
                for tool_call in msg.tool_calls:
                    print(f"\n🔧 呼叫工具: {tool_call['name']}")
                    print(f"   參數：{preview_text(tool_call['args'], 120)}")
            elif isinstance(msg, ToolMessage):
                # 工具回傳通常很長，只截前 150 字供確認，避免洗版
                print(f"✅ 工具回傳: {preview_text(msg.content, 150)}", flush=True)


def _print_preferences(output):
    """偏好前置檢索完成：印出檢索到的偏好片段摘要。"""
    preferences = output.get("preferences")
    if not preferences:
        return
    print_header("Retrieve Preferences — 偏好前置檢索")
    print('📚 輸出欄位：state["preferences"]（偏好原文片段）')
    print(preview_text(preferences, 600))


def _print_plan(output):
    """計畫產出完成：逐條列出步驟。"""
    print_header("Planner — 規劃 / 修訂計畫")
    print('輸出欄位：state["plan"]（可修改的計畫物件）')
    for i, step in enumerate(output.get("plan", []), 1):
        print(f"  {i}. {step}")


def _print_critique(output):
    """審核完成：印出 verdict、修正次數與發現的問題。"""
    critique = output.get("critique")
    print_header("Reflect — 多面向品質檢查")
    print('輸出欄位：state["critique"]（structured output）')
    print(f"  verdict：{critique['verdict'] if critique else '?'}")
    print(f"  revisions：{output.get('revisions', 0)}")
    issues = critique["issues"] if critique else []
    if issues:
        print("  issues：")
        for issue in issues:
            print(f"    ・{issue}")


# 頂層節點名稱 → 對應的顯示函式
NODE_PRINTERS = {
    "retrieve_preferences": _print_preferences,
    "planner": _print_plan,
    "reflect": _print_critique,
}


def _handle_event(ns, mode, data, st: dict):
    """依事件來源（頂層/子圖）與類型（messages/updates）分流顯示。"""
    # ── 子圖（executor 內部）事件 ──
    if ns:
        if mode == "updates":
            _print_tool_activity(data)
            return
        chunk, _meta = data
        # 只串流 LLM 的推理/行程文字；工具回傳內容交給 _print_tool_activity 截斷顯示
        if not chunk.content or isinstance(chunk, ToolMessage):
            return
        if st["header"] != "executor":
            print_header("Executor — 執行與生成")
            st["header"] = "executor"
        print(chunk.content, end="", flush=True)
        return

    # ── 頂層事件 ──
    # planner/reflect 走 structured output（function_calling），沒有逐 token 文字可串流
    if mode == "messages":
        return

    # mode == "updates"：節點跑完時印出寫入 state 的欄位
    for node_name, output in data.items():
        printer = NODE_PRINTERS.get(node_name)
        if printer:
            printer(output)
    # 任何頂層節點跑完都重置 header；executor 串流時才會重新印標題
    st["header"] = None


async def run_chat(graph):
    """啟動多輪對話介面，直到使用者主動結束。"""
    # 設定固定 thread_id，讓 graph 可以保留多輪對話記憶
    config = {"configurable": {"thread_id": "travel-session-1"}}
    # 串流狀態：跨事件記住目前 header，避免同一節點重複印標題
    stream_state = {"header": None}

    print(
        """
==================================================
🧳 個人化旅遊規劃 Agentic AI（LangGraph）已就緒
💡 輸入旅遊需求開始規劃，輸入 'exit' 或 'quit' 結束
💡 範例：
   1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
   2. 幫我把 Day 2 改成以室內景點為主
==================================================
"""
    )

    # 持續接收使用者輸入，直到 read_query 回傳 None
    turn = 1
    while True:
        user_input = read_query(turn)
        if user_input is None:
            print("\n👋 再見")
            break
        if not user_input:
            continue

        # 本輪輸入：丟新的 HumanMessage（add_messages 自動 append），
        # 並重置 transient 欄位，避免上一輪殘留干擾
        payload = {
            "messages": [HumanMessage(content=user_input)],
            "plan": [],
            "critique": None,
            "revisions": 0,
        }

        # 串流執行本輪 graph，依事件即時顯示各節點進度
        stream_state["header"] = None
        async for ns, mode, data in graph.astream(
            payload, config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            _handle_event(ns, mode, data, stream_state)

        print("\n\n✅ 本輪規劃完成\n")
        turn += 1
