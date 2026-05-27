"""
nodes.py — StateGraph 節點函式模組

定義 LangGraph StateGraph 中的六個節點：
  1. plan_node      — 任務拆解（Planning）
  2. retrieve_node  — 偏好檢索（RAG）
  3. search_node    — 資訊蒐集（MCP tools）
  4. generate_node  — 行程生成
  5. reflect_node   — 可行性評估（Reflection + Proactivity）
  6. respond_node   — 回覆使用者

每個節點接收 TravelState，回傳要更新的欄位（partial update）。
使用工廠函式 create_nodes() 將 LLM、檢索器、工具注入各節點。
"""

from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state import TravelState
from rag import build_preference_query


# ──────────────────────────────────────────────
# 各節點的 System Prompt
# ──────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """你是一位旅遊規劃的任務分析師。
根據使用者的旅遊需求，將其拆解為具體的子任務清單。
子任務應涵蓋：偏好分析、景點搜尋、天氣查詢、匯率查詢、預算估算等面向。
只輸出子任務清單，不要開始執行任何任務。"""

SEARCH_SYSTEM_PROMPT = """你是一位旅遊資訊蒐集助理。
你的任務只負責蒐集外部資料，不負責規劃每日行程。
請根據使用者的旅遊需求，使用可用工具蒐集以下資訊：
1. 使用 tavily_search 搜尋目的地的景點、古蹟、美食、住宿、交通等旅遊資訊（至少搜尋 2 次，每次只查一個主題）
2. 使用天氣查詢工具查詢旅行期間的天氣預報。若旅行日期超出預設預報範圍，請主動帶入 start_date / end_date 參數重試
3. 使用 Frankfurter MCP 的 convert 或 get_rates 查詢台幣兌換當地貨幣的匯率；必要時先用 list_currencies 確認可用貨幣代碼

請依序完成查詢，並將所有結果整理為外部資訊摘要，固定包含以下小節（用純文字標題即可）：
景點/古蹟資訊
天氣資訊
匯率資訊
交通資訊
住宿候選資訊
查詢不足或失敗

輸出格式要求：
- 全程純文字，禁止任何 markdown 語法
- 禁止使用：# ## ### 標題符號、**粗體**、*斜體*、`程式碼`、表格、---- 分隔線、> 引言、emoji
- 條列請用「・」或「1. 2. 3.」開頭，不要用「-」「*」
- 章節標題直接寫文字並換行，不要加任何符號裝飾

禁止輸出完整三天行程、Day 1/Day 2/Day 3 每日排程、總預算結論或最終旅遊建議。
這些內容留給 Generate 節點處理。
工具查詢完成後一定要輸出文字摘要，不可以回覆空白。"""

GENERATE_SYSTEM_PROMPT = """你是一位專業的個人化旅遊規劃助理。
你的任務是根據使用者的旅遊偏好與即時資訊，產出完整的旅遊行程規劃。

任務流程：
1. 先從過往旅行紀錄歸納使用者的旅行風格
2. 以三段式格式說明使用者偏好：
   - 原文依據：引用旅行紀錄中的具體描述
   - 推理結果：從原文推斷出的偏好特徵
   - 行程影響：該偏好如何影響本次行程安排
3. 根據偏好與即時資訊，規劃每日行程（含上午、午餐、下午、晚上時段）
4. 標註各景點的預計費用（以當地貨幣和台幣分別列出）
5. 提供預算總計、住宿推薦、交通建議與注意事項

輸出格式要求：
- 全程使用純文字，禁止任何 markdown 語法
- 禁止使用：# ## ### 標題符號、**粗體**、*斜體*、`程式碼`、表格、---- 分隔線、> 引言、emoji
- 條列請用「・」或「1. 2. 3.」純文字符號，不要用「-」或「*」開頭
- 預算明細用條列格式逐項列出，禁止用空格或 Tab 對齊欄位（會形成偽表格）
  範例：・機票（台北-大阪來回）：約 30,000 日圓（約 6,000 台幣）
- 章節標題直接寫文字並換行，不要加任何符號裝飾
- 費用以實際匯率換算

輸出長度限制：
- 整體控制在 1800 字以內
- 偏好分析三段式：每段「原文依據」「推理結果」「行程影響」各限 1 句話
- 每日行程每個時段（上午/午餐/下午/晚上）以 2 句話內描述完畢，不要長篇說明
- 預算、住宿、交通、注意事項合計不超過 400 字
- 不要重複資訊（例如門票費用已列在景點段就不要再列在預算段細項）"""

REFLECT_SYSTEM_PROMPT = """你是一位嚴謹的旅遊行程審核員。
請客觀評估行程草案的可行性，重點檢查以下項目：
1. 預算合理性：使用實際匯率計算，確認總花費是否在合理範圍內
2. 偏好符合度：景點安排是否符合使用者過往旅遊風格
3. 天氣適配性：是否考慮天氣影響，雨天是否有室內備案
4. 動線合理性：每日景點之間的交通動線是否順暢

評估結果：
- 如果行程通過評估，請在回覆第一行輸出 PASS，接著簡述通過原因。
- 如果需要修正，請在回覆第一行輸出 REVISE，接著具體說明需要修正的項目與建議方向。

輸出格式要求：
- 全程使用純文字，禁止任何 markdown 語法
- 禁止使用：# ## ### 標題符號、**粗體**、*斜體*、`程式碼`、表格、---- 分隔線、> 引言、emoji
- 條列請用「・」或「1. 2. 3.」開頭，不要用「-」或「*」"""


def _preview_tool_value(value, max_chars: int = 300) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...（共 {len(text)} 字元）"


def create_nodes(llm, retriever, tools):
    """
    工廠函式：注入 LLM、檢索器與工具，回傳所有節點函式。

    Parameters:
        llm: ChatOpenAI 實例（NVIDIA API）
        retriever: Milvus RAG 檢索器
        tools: 完整工具清單（MCP tools）
    Returns:
        dict: 節點名稱 → 節點函式 的對應表
    """
    tool_map = {t.name: t for t in tools}

    # ──────────────────────────────────────────
    # 節點 1：Plan（任務拆解）
    # ──────────────────────────────────────────
    async def plan_node(state: TravelState) -> dict:
        messages = [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"[對話記錄]\n{state['chat_history'] or '無'}\n\n"
                f"[本次使用者旅遊需求]\n{state['user_query']}"
            )),
        ]
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)
        return {"plan": "".join(chunks)}

    # ──────────────────────────────────────────
    # 節點 2：Retrieve（偏好檢索）
    # ──────────────────────────────────────────
    async def retrieve_node(state: TravelState) -> dict:
        query = build_preference_query(state["user_query"])
        docs = retriever.invoke(query)
        preferences = "\n\n".join(doc.page_content for doc in docs)
        return {"preferences": preferences}

    # ──────────────────────────────────────────
    # 節點 3：Search（資訊蒐集）
    # ──────────────────────────────────────────
    async def search_node(state: TravelState) -> dict:
        llm_with_tools = llm.bind_tools(tools)
        tool_results = []
        today = datetime.now().strftime("%Y-%m-%d")

        messages = [
            SystemMessage(content=SEARCH_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"今日日期：{today}\n"
                    f"[對話記錄]\n{state['chat_history'] or '無'}\n\n"
                    f"[本次使用者旅遊需求]\n{state['user_query']}\n"
                    f"[任務計畫]\n{state['plan']}"
                )
            ),
        ]

        # 工具呼叫迴圈（最多 8 輪），用 streaming 避免長輸出觸發 504
        for _ in range(8):
            response = None
            async for chunk in llm_with_tools.astream(messages):
                response = chunk if response is None else response + chunk
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                print("\n[Tool Call]")
                print(f"工具名稱：{tc['name']}")
                print(f"工具參數：{_preview_tool_value(tc['args'])}")

                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    result = f"工具 {tc['name']} 不存在"
                else:
                    try:
                        result = await tool_fn.ainvoke(tc["args"])
                    except Exception as e:
                        result = f"工具呼叫失敗：{e}"

                print(f"工具結果：{_preview_tool_value(result)}", flush=True)

                tool_results.append({"name": tc["name"], "result": str(result)})
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        external_info = response.content.strip()
        if not external_info and tool_results:
            external_info = "\n\n".join(
                f"[{r['name']}]\n{r['result']}" for r in tool_results
            )
        return {"external_info": external_info or "外部資訊查詢未取得結果"}

    # ──────────────────────────────────────────
    # 節點 4：Generate（行程生成）
    # ──────────────────────────────────────────
    async def generate_node(state: TravelState) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")

        # 如果是重新規劃，附上前次反思意見
        revision_note = (
            f"\n\n[前次評估意見（請針對以下問題修正）]\n{state['reflection']}"
            if state["reflection"] else ""
        )

        prompt = (
            f"[今日日期：{today}]\n\n"
            f"[使用者旅遊需求]\n{state['user_query']}\n\n"
            f"[對話記錄]\n{state['chat_history'] or '無'}\n\n"
            f"[使用者旅遊偏好（來自過往旅遊紀錄）]\n{state['preferences']}\n\n"
            f"[即時資訊（景點、天氣、匯率）]\n{state['external_info']}"
            f"{revision_note}"
        )

        messages = [
            SystemMessage(content=GENERATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        # 使用 streaming 避免 NVIDIA 網關 504 timeout（長輸出會被砍）
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)
        return {"draft_itinerary": "".join(chunks)}

    # ──────────────────────────────────────────
    # 節點 5：Reflect（可行性評估）
    # ──────────────────────────────────────────
    async def reflect_node(state: TravelState) -> dict:
        prompt = (
            f"[行程草案]\n{state['draft_itinerary']}\n\n"
            f"[使用者原始需求]\n{state['user_query']}\n\n"
            f"[即時資訊（天氣與匯率）]\n{state['external_info']}"
        )

        messages = [
            SystemMessage(content=REFLECT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)
        content = "".join(chunks)
        is_feasible = content.strip().upper().startswith("PASS")

        return {
            "reflection": content,
            "is_feasible": is_feasible,
            "revision_count": state["revision_count"] + 1,
        }

    # ──────────────────────────────────────────
    # 節點 6：Respond（回覆使用者）
    # ──────────────────────────────────────────
    async def respond_node(state: TravelState) -> dict:
        return {"final_response": state["draft_itinerary"]}

    # 回傳所有節點
    return {
        "plan": plan_node,
        "retrieve": retrieve_node,
        "search": search_node,
        "generate": generate_node,
        "reflect": reflect_node,
        "respond": respond_node,
    }
