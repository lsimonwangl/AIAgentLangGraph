"""
chat.py — 對話迴圈與串流輸出模組

升級為 Agentic 架構後，新增 Brain 節點的決策顯示：
每次 brain 跑完都會印出「決策：執行 X 動作（理由：Y）」，
讓使用者看到 agent 的思考軌跡，而不是只有節點輸出。
"""

from langchain_core.messages import HumanMessage

from prompt import read_query
from state import TravelState

# 節點名稱對應的中文標題
NODE_TITLES = {
    "brain": "Brain — 下一步決策",
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

    if node_name == "retrieve":
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

    elif node_name == "respond":
        print_state_field(
            "final_response",
            "Respond 節點最後回覆給使用者的內容。",
            output.get("final_response", ""),
            max_chars=2400,
        )

    elif node_name == "brain":
        action = output.get("next_action", "")
        reasoning = output.get("brain_reasoning", "")
        step = output.get("step_count", 0)
        print(f"步驟 #{step}：決定執行 [{action}]")
        print(f"理由：{reasoning}")


async def chat_loop(graph):
    """
    啟動多輪對話迴圈。

    每輪接收使用者輸入，交給 StateGraph 執行：由 brain 動態決定
    Retrieve / Search / Generate / Reflect 的執行順序，最後 Respond 回覆，
    並以 stream 逐節點顯示執行進度與 brain 的決策軌跡。
    """
    config = {"configurable": {"thread_id": "travel-session-1"}}

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

        # 本輪輸入：只塞新的 HumanMessage（由 add_messages reducer 自動 append
        # 到既有 messages），並重置所有 transient 欄位，避免上一輪殘留干擾本輪。
        input_state: TravelState = {
            "messages": [HumanMessage(content=user_input)],
            "preferences": "",
            "external_info": "",
            "draft_itinerary": "",
            "reflection": "",
            "revision_count": 0,
            "final_response": "",
            # Brain 控制欄位：每輪重置，由 brain 自己累積
            "next_action": "",
            "brain_reasoning": "",
            "step_count": 0,
        }

        # 逐節點執行：LLM 節點以 token 即時顯示，非 LLM 節點顯示完整輸出
        streaming_nodes = {"search", "generate", "reflect"}
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
                        # 內容已串流完，reflect 補印評估結果（PASS / REVISE / RESEARCH）
                        if node_name == "reflect":
                            verdict = output.get("reflection", "").strip().split("\n")[0][:20]
                            print(f"\n\n[評估結果] {verdict}")
                        current_stream_node = None
                    else:
                        print_node_output(node_name, output)

        # 對話記錄已由 add_messages reducer + MemorySaver checkpointer 自動保存，
        # 下一輪只要丟新的 HumanMessage 進來，過往訊息自然會被帶入。
