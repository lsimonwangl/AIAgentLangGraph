"""
chat.py — 對話迴圈與串流輸出模組

對應 Lab1 Dify 的直接回覆節點與 Lab2 的 chat.py。

相較於 Lab2 直接串流 Agent 的回答，Lab4 改為逐節點顯示執行狀態，
讓使用者清楚看到 StateGraph 各節點的執行順序與產出，
特別是 Reflect 節點觸發重新規劃時的回饋迴圈。
"""

from prompt import read_query
from state import TravelState

# 節點名稱對應的中文標題
NODE_TITLES = {
    "plan": "Plan — 任務拆解",
    "retrieve": "Retrieve — 偏好檢索",
    "search": "Search — 資訊蒐集",
    "generate": "Generate — 行程生成",
    "reflect": "Reflect — 可行性評估",
    "respond": "Respond — 回覆使用者",
}


def preview_text(text: str, max_chars: int = 1200) -> str:
    """回傳適合終端機顯示的內容預覽，避免長文字刷滿畫面。"""
    text = text.strip()
    if not text:
        return "（空）"
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n...（已截斷，完整內容共 {len(text)} 字元）"


def print_state_field(field_name: str, description: str, value, max_chars: int = 1200):
    """以教學模式顯示節點輸出的 State 欄位。"""
    print(f"輸出欄位：state[\"{field_name}\"]")
    print(f"欄位用途：{description}")
    print("內容預覽：")
    print(preview_text(str(value), max_chars=max_chars))


def print_node_output(node_name: str, output: dict):
    """根據節點類型，格式化並印出對應的執行結果"""
    title = NODE_TITLES.get(node_name, node_name)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    if node_name == "plan":
        print_state_field(
            "plan",
            "Plan 節點產生的任務拆解，供後續 Search 節點決定要查什麼。",
            output.get("plan", ""),
        )

    elif node_name == "retrieve":
        prefs = output.get("preferences", "")
        line_count = prefs.count("\n") + 1
        print(f"檢索摘要：從 Milvus 知識庫檢索到 {line_count} 行偏好資料（共 {len(prefs)} 字元）")
        print()
        print_state_field(
            "preferences",
            "Retrieve 節點從 RAG 知識庫找出的使用者過往旅遊偏好。",
            prefs,
        )

    elif node_name == "search":
        info = output.get("external_info", "")
        print(f"蒐集摘要：已蒐集 {len(info)} 字元的外部資訊（景點、天氣、匯率）")
        print()
        print_state_field(
            "external_info",
            "Search 節點透過 MCP tools 蒐集到的即時外部資訊。",
            info,
        )

    elif node_name == "generate":
        print_state_field(
            "draft_itinerary",
            "Generate 節點根據使用者偏好與外部資訊產生的行程草案。",
            output.get("draft_itinerary", ""),
        )

    elif node_name == "reflect":
        is_feasible = output.get("is_feasible", False)
        count = output.get("revision_count", 0)
        print_state_field(
            "is_feasible",
            "Reflect 節點判斷行程是否通過可行性評估。",
            is_feasible,
            max_chars=200,
        )
        print()
        print_state_field(
            "revision_count",
            "Reflect 節點累計已評估幾次，用來避免無限修正。",
            count,
            max_chars=200,
        )
        print()
        print_state_field(
            "reflection",
            "Reflect 節點對行程草案的審核意見；第一行 PASS/REVISE 會影響下一個節點。",
            output.get("reflection", ""),
        )

    elif node_name == "respond":
        print_state_field(
            "final_response",
            "Respond 節點最後回覆給使用者的內容。",
            output.get("final_response", ""),
            max_chars=2400,
        )


async def chat_loop(graph):
    """
    啟動多輪對話迴圈。

    每輪接收使用者輸入，透過 StateGraph 完整執行
    Plan → Retrieve → Search → Generate → Reflect → Respond 流程，
    並以 stream_mode="updates" 逐節點顯示執行進度。
    """
    config = {"configurable": {"thread_id": "travel-session-1"}}
    chat_history = ""

    print("\n" + "="*60)
    print("  個人化旅遊規劃 Agentic AI（LangGraph）")
    print("  輸入旅遊需求開始規劃，輸入 quit 結束")
    print("="*60)

    while True:
        user_input = read_query()

        if user_input.lower() == "quit":
            print("\n感謝使用，再見！")
            break

        if not user_input:
            continue

        # 初始化 State
        input_state: TravelState = {
            "user_query": user_input,
            "chat_history": chat_history,
            "plan": "",
            "preferences": "",
            "external_info": "",
            "draft_itinerary": "",
            "reflection": "",
            "is_feasible": False,
            "revision_count": 0,
            "final_response": "",
        }

        # 逐節點執行：LLM 節點以 token 即時顯示，非 LLM 節點顯示完整輸出
        streaming_nodes = {"plan", "search", "generate", "reflect"}
        current_stream_node = None
        async for mode, payload in graph.astream(
            input_state, config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = payload
                node = meta.get("langgraph_node")
                if node not in streaming_nodes or not chunk.content:
                    continue
                if node != current_stream_node:
                    print(f"\n{'='*60}\n  {NODE_TITLES.get(node, node)}\n{'='*60}")
                    current_stream_node = node
                print(chunk.content, end="", flush=True)
            else:  # mode == "updates"
                for node_name, output in payload.items():
                    if node_name in streaming_nodes:
                        # 內容已串流完，補印附屬欄位（reflect 的 is_feasible / revision_count）
                        if node_name == "reflect":
                            print(
                                f"\n\n[is_feasible] {output.get('is_feasible')}"
                                f"  [revision_count] {output.get('revision_count')}"
                            )
                        current_stream_node = None
                    else:
                        print_node_output(node_name, output)

        # 取得最終結果，更新對話記錄
        final_state = await graph.aget_state(config)
        final_response = final_state.values.get("final_response", "")
        chat_history += f"\n使用者：{user_input}\n助理：{final_response}\n"
