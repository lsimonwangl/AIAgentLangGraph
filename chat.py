"""
Travel Agent - 終端機輸出
=======================
chat.py 負責顯示啟動畫面、串流各節點執行進度，並驅動多輪對話迴圈。

程式流程：
  1. 顯示啟動畫面與旅遊問題範例。
  2. 從終端機接收使用者輸入。
  3. 使用 graph.astream() 接收頂層節點與 Executor 子圖事件。
  4. 依事件類型顯示偏好、計畫、工具活動、行程草案與審核結果。
  5. NVIDIA API 呼叫失敗時只中止該輪，讓使用者稍後重新提問。
  6. 保留同一個 thread_id，持續處理多輪對話直到使用者結束。

串流用 stream_mode=["updates","messages"] + subgraphs=True：
    - messages：即時 token（executor 行程草案；planner/reflect 是 structured output，無文字可串）
    - updates ：節點產出的 state 欄位（preferences / plan / critique），與子圖工具呼叫

"""

# ── 載入套件 ──────────────────────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import APIError

from prompt import read_query


# ── 終端機文字顯示輔助函式 ────────────────────────────────
def preview_text(text: str, max_chars: int = 400) -> str:
    """截斷過長文字，避免工具回傳洗版終端機。"""
    # 將各種輸入統一轉成字串，並移除前後空白。
    text = str(text).strip()
    # 空內容使用固定提示，避免終端畫面看起來像漏印資料。
    if not text:
        return "（空）"
    # 文字未超過上限時直接完整回傳。
    if len(text) <= max_chars:
        return text
    # 超過上限時只保留開頭，並標示原始總字數。
    return f"{text[:max_chars].rstrip()}...（共 {len(text)} 字元）"


def print_header(title: str):
    """印出節點區段標題，讓使用者知道 graph 跑到哪個階段。"""
    # 前後各印一條 60 字元分隔線，讓不同節點的輸出容易辨識。
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ── 顯示 Executor 工具呼叫與回傳結果 ─────────────────────
def _print_tool_activity(node_output: dict):
    """顯示 executor 子圖的工具呼叫與回傳結果。

    executor 是 create_agent 子圖，其工具呼叫/結果走 subgraph 的 updates 事件。
    """
    # 子圖更新資料的 key 是內部節點名稱，因此逐一檢查所有 value。
    for value in node_output.values():
        # 某些更新不是 dict，也就不會包含 messages 欄位，直接略過。
        if not isinstance(value, dict):
            continue
        # 取出這個子圖節點本次新增的所有訊息。
        for msg in value.get("messages", []):
            # 帶有 tool_calls 的 AIMessage 代表 Agent 正準備呼叫一個或多個工具。
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 同一則 AIMessage 可能要求多個工具，因此逐一顯示名稱與參數。
                for tool_call in msg.tool_calls:
                    # 先顯示工具名稱，讓使用者知道 Agent 正在執行哪種查詢。
                    print(f"\n🔧 呼叫工具: {tool_call['name']}")
                    # 工具參數可能很長，最多顯示 120 個字元。
                    print(f"   參數：{preview_text(tool_call['args'], 120)}")
            # ToolMessage 代表外部工具已經執行完成並回傳結果。
            elif isinstance(msg, ToolMessage):
                # 工具內容通常很長，只顯示前 150 個字元供使用者確認。
                print(f"✅ 工具回傳: {preview_text(msg.content, 150)}", flush=True)


# ── 顯示頂層節點輸出 ──────────────────────────────────────
def _print_preferences(output):
    """偏好前置檢索完成：印出檢索到的偏好片段摘要。"""
    # 從節點更新資料取出寫入 State 的 preferences 欄位。
    preferences = output.get("preferences")
    # 沒有命中偏好資料時不印空區段，讓終端畫面保持簡潔。
    if not preferences:
        return
    # 顯示目前完成的節點名稱。
    print_header("Retrieve Preferences — 偏好前置檢索")
    # 明確標出這段內容對應的 State 欄位，方便教學時對照程式。
    print('📚 輸出欄位：state["preferences"]（偏好原文片段）')
    # 偏好原文可能較長，最多顯示 600 個字元。
    print(preview_text(preferences, 600))


def _print_plan(output):
    """計畫產出完成：逐條列出步驟。"""
    # 顯示 Planner 節點標題與它更新的 State 欄位。
    print_header("Planner — 規劃 / 修訂計畫")
    print('輸出欄位：state["plan"]（可修改的計畫物件）')
    # enumerate() 從 1 開始編號，依序印出 Plan.steps 的每個步驟。
    for i, step in enumerate(output.get("plan", []), 1):
        print(f"  {i}. {step}")


def _print_critique(output):
    """審核完成：印出 verdict、審核輪次與發現的問題。"""
    # 從節點更新資料取出 Reflect 產生的結構化審核結果。
    critique = output.get("critique")
    # 顯示 Reflect 節點標題與它更新的 State 欄位。
    print_header("Reflect — 多面向品質檢查")
    print('輸出欄位：state["critique"]（structured output）')
    # critique 不存在時以 ? 代替，避免終端顯示程式錯誤。
    print(f"  verdict：{critique['verdict'] if critique else '?'}")
    # revisions 表示目前已執行的審核輪次。
    print(f"  revisions：{output.get('revisions', 0)}")
    # 通過時 issues 通常為空 list；沒有 critique 時同樣使用空 list。
    issues = critique["issues"] if critique else []
    # 只有確實存在問題時才顯示 issues 標題與內容。
    if issues:
        print("  issues：")
        # 每個 issue 獨立一行，方便觀察 Planner 下一輪要修改什麼。
        for issue in issues:
            print(f"    ・{issue}")


# 將頂層節點名稱對應到各自的顯示函式，避免在事件處理器寫多層 if/elif。
NODE_PRINTERS = {
    "retrieve_preferences": _print_preferences,  # 顯示偏好檢索結果
    "planner": _print_plan,  # 顯示首次規劃或修訂計畫
    "reflect": _print_critique,  # 顯示結構化審核結果
}


# ── 依事件來源與類型分流顯示 ──────────────────────────────
def _handle_event(ns, mode, data, st: dict):
    """依事件來源（頂層/子圖）與類型（messages/updates）分流顯示。"""
    # ── 子圖（executor 內部）事件 ──
    # ns 有內容代表事件來自 create_agent 建立的 Executor 子圖。
    if ns:
        # updates 模式包含工具呼叫與工具回傳，交給專用函式顯示。
        if mode == "updates":
            _print_tool_activity(data)
            # 工具活動處理完成後直接返回，不再當成 token 串流處理。
            return
        # messages 模式的 data 是「訊息片段、metadata」二元素組。
        chunk, _meta = data
        # 沒有文字內容的片段不需顯示；ToolMessage 已在 updates 模式截斷顯示。
        if not chunk.content or isinstance(chunk, ToolMessage):
            return
        # 第一次收到 Executor 文字時才印標題，避免每個 token 都重複印一次。
        if st["header"] != "executor":
            print_header("Executor — 執行與生成")
            # 記錄標題已顯示，後續 token 直接接續輸出。
            st["header"] = "executor"
        # end="" 讓 token 連續接在同一段文字，flush=True 則立即更新終端畫面。
        print(chunk.content, end="", flush=True)
        return

    # ── 頂層事件 ──
    # Planner 與 Reflect 使用 structured output，不需要處理頂層 messages token。
    if mode == "messages":
        return

    # updates 模式的 key 是剛完成的頂層節點名稱，value 是它寫入 State 的欄位。
    for node_name, output in data.items():
        # 依節點名稱查找對應的顯示函式；Executor 沒有對應項目，會得到 None。
        printer = NODE_PRINTERS.get(node_name)
        # 只有需要額外顯示的頂層節點才呼叫 printer。
        if printer:
            printer(output)
    # 頂層節點完成後重置標題狀態，下一輪 Executor 串流時會重新印出標題。
    st["header"] = None


# ── 啟動多輪終端機對話 ────────────────────────────────────
async def run_chat(graph):
    """啟動多輪對話介面，直到使用者主動結束。"""
    # 固定 thread_id 讓 MemorySaver 把每輪輸入接在同一段旅遊對話中。
    config = {"configurable": {"thread_id": "travel-session-1"}}
    # 這個 dict 跨多個串流事件共用，用來記住 Executor 標題是否已經顯示。
    stream_state = {"header": None}

    # 顯示程式啟動訊息、結束方式與兩個多輪對話範例。
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

    # 第一輪從 1 開始顯示，之後每完成一次規劃就加一。
    turn = 1
    # 持續接收輸入，直到使用者輸入結束關鍵字或按下 Ctrl+C / Ctrl+D。
    while True:
        # 呼叫 prompt.py 顯示目前輪次並讀取一行輸入。
        user_input = read_query(turn)
        # None 代表使用者要求結束對話。
        if user_input is None:
            print("\n👋 再見")
            break
        # 空字串代表使用者只按 Enter，不執行 Graph 並繼續等待輸入。
        if not user_input:
            continue

        # 建立本輪 Graph 輸入；messages 使用 MessagesState 的 reducer 自動附加到歷史。
        # plan、critique 與 revisions 屬於單輪暫存資料，因此每輪開始時重置。
        payload = {
            "messages": [HumanMessage(content=user_input)],  # 本輪新的使用者需求
            "plan": [],  # 清除上一輪執行計畫
            "critique": None,  # 清除上一輪審核結果
            "revisions": 0,  # 審核輪次重新從 0 開始
        }

        # 新一輪開始前清除 Executor 標題狀態。
        stream_state["header"] = None
        # 將整輪 Graph 執行包在 try/except，避免 NVIDIA API 暫時過載時終止整個程式。
        try:
            # 同時訂閱 updates 與 messages，並開啟 subgraphs 取得 Executor 內部工具事件。
            async for ns, mode, data in graph.astream(
                payload, config, stream_mode=["updates", "messages"], subgraphs=True
            ):
                # 每收到一個事件就依來源與類型立即更新終端畫面。
                _handle_event(ns, mode, data, stream_state)
        # 只捕捉模型端點回傳的 API 錯誤；其他程式錯誤仍保留 traceback，方便除錯。
        except APIError as error:
            print(f"\n\n⚠️ NVIDIA API 呼叫失敗，這輪規劃未完成：{error}")
            print("   通常是端點限流或服務暫時過載，稍等一兩分鐘再重問一次即可。\n")
        else:
            # astream 正常結束代表本輪已通過審核或達到最大審核輪次。
            print("\n\n✅ 本輪規劃完成\n")

        # 下一次輸入顯示新的輪次編號。
        turn += 1
