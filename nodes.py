"""
nodes.py — Agentic 節點函式模組

本版本由「線性 workflow」升級為「中央 brain 路由」架構：
  0. brain_node     — 決策中樞，每次決定下一步要執行哪個動作
  1. retrieve_node  — 偏好檢索（RAG）
  2. search_node    — 資訊蒐集（MCP tools）
  3. generate_node  — 行程生成
  4. reflect_node   — 可行性評估（Reflection）
  5. respond_node   — 回覆使用者

執行流程：brain → action → brain → action → ... → finish → respond → END
brain 依照目前 State 決定下一步，沒有寫死的順序。
"""

from datetime import datetime
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from state import TravelState
from rag import build_preference_query


class BrainDecision(BaseModel):
    """Brain 對下一步動作的結構化決策。"""

    reasoning: str = Field(description="選擇此動作的簡短理由（一句話）")
    action: Literal[
        "retrieve", "search", "generate", "finish"
    ] = Field(description="下一步要執行的動作名稱")


def _get_user_query(messages: list[BaseMessage]) -> str:
    """取出最新一則使用者輸入，作為本輪的 user_query。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _format_history(messages: list[BaseMessage]) -> str:
    """將過往對話訊息整理成可塞進 prompt 的字串，排除最新一輪使用者輸入。"""
    if len(messages) <= 1:
        return "無"
    lines = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            lines.append(f"使用者：{msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"助理：{msg.content}")
    return "\n".join(lines) if lines else "無"


# ──────────────────────────────────────────────
# 各節點的 System Prompt
# ──────────────────────────────────────────────

BRAIN_SYSTEM_PROMPT = """你是個人化旅遊規劃 Agent 的大腦，負責決定下一步要做什麼。

可選動作：
・retrieve — 從使用者過往旅遊紀錄 RAG 檢索偏好
・search — 用 MCP 工具蒐集景點、天氣、匯率等外部資訊
・generate — 根據偏好與外部資訊產出完整行程（產出後系統會「自動」評估 reflect，你不必也不能選 reflect）
・finish — 結束流程，把目前草案回覆使用者

注意：reflect 不是你的選項，它由 generate 自動觸發。所以你只會在
retrieve / search / reflect 完成後被詢問下一步，「上一個動作」不會是 generate。

決策原則（依優先順序檢查）：
1. 若「上一個動作」是 reflect 且結果為 PASS → finish
2. 若「已生成 N 版」中 N >= 5 → finish（修正已達上限）
3. 沒有偏好資料 → retrieve
4. 沒有外部資訊（天氣/匯率/景點）→ search
5. 已有偏好與外部資訊、但尚未產出草案 → generate（產出後會自動 reflect）
6. 若「上一個動作」是 reflect，依評估結果分流：
   - 結果為 RESEARCH（資訊不足／有誤）→ search 回去補查缺的資訊
   - 結果為 REVISE 且「已生成 N 版」< 5 → generate 修正（會再自動 reflect）
7. 簡單追問（例如「那住宿建議哪間？」）若 history 已有資訊 → 直接 finish
8. 同一動作不要無謂重複，除非有明確理由（例如 RESEARCH 重查、依 reflection 修正）

只輸出 reasoning（一句話，必須引用「上一個動作」與相關狀態）和 action 兩個欄位。"""

SEARCH_SYSTEM_PROMPT = """你是一位旅遊資訊蒐集助理。
你的任務只負責蒐集外部資料，不負責規劃每日行程。
請根據使用者的旅遊需求，使用可用工具蒐集以下資訊：
1. 使用 tavily_search 搜尋目的地的景點、美食、住宿、交通（最多 3 次搜尋，每次主題不同）
2. 使用天氣查詢工具查詢旅行期間天氣預報；超出預設範圍時帶入 start_date / end_date 重試
3. 使用 Frankfurter MCP 的 convert 查詢匯率

搜尋時請參考「使用者偏好」做針對性查詢，而非只查通用熱門結果。例如：
偏好設計感／老屋改建住宿 → 住宿關鍵字帶入「設計旅店」「老屋改建」等；
偏好在地小吃 → 美食往庶民老店方向查；
偏好特定景點類型（如有庭園、自然元素的古蹟）→ 景點搜尋也朝該方向。
這樣查回的資料才貼合使用者，後續行程才不需臨時編造。

若這次是因 RESEARCH 回來補查，請優先處理審核指出的缺口：
- 缺住宿價格 → 搜尋具體飯店名稱 + 房價/price/rate，或查同區同類設計旅店的合理價格範圍
- 缺餐飲價格 → 搜尋具體店家或餐飲區域的人均價格/菜單價格/平均消費
- 若補查後仍找不到可靠來源，摘要中要明確寫「價格未查得可靠來源，需以訂房網站為準」
  或「價格未查得可靠來源，需以現場為準」，不要自行編成確定價格。

請將所有結果整理為精簡摘要，固定包含以下小節：
景點資訊
天氣資訊
匯率資訊
交通資訊
住宿候選

摘要規則：
- 整體控制在 800 字以內
- 每個景點/飯店一行：名稱、價格/門票、地鐵站、一句特色
  範例：・大阪城天守閣 1,200日圓 谷町四丁目站 日本三大名城
- 飯店或餐飲價格若無可靠來源，一行內直接標註「價格未查得可靠來源」
- 天氣只列日期、最高/最低溫、降雨機率、雨量，不要長篇敘述
- 匯率一行寫完
- 禁止寫景點歷史背景、文化典故、行程建議
- 禁止任何 markdown 語法（# ** > 表格 ---- emoji 等）
- 條列用「・」或「1. 2. 3.」開頭

禁止輸出完整三天行程、Day 1/Day 2/Day 3 排程、總預算或最終建議——留給 Generate 節點。
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

價格使用規則：
- Search 資訊有明確價格時，才可寫成確定價格，例如「8,000日圓」
- Search 資訊沒有住宿價格時，如仍需估預算，必須寫成「估算：約X日圓，實際以訂房網站為準」
- Search 資訊沒有餐飲價格時，如仍需估預算，必須寫成「估算：約X日圓，實際以現場為準」
- 禁止把估算價格寫成已查證價格；任何估算都必須出現「估算」二字
- 使用周遊卡或套票已涵蓋的門票，不要在總預算中重複計入單點門票

輸出格式要求：
- 全程使用純文字，禁止任何 markdown 語法
- 禁止使用：# ## ### 標題符號、**粗體**、*斜體*、`程式碼`、表格、---- 分隔線、> 引言、emoji
- 條列請用「・」或「1. 2. 3.」純文字符號，不要用「-」或「*」開頭
- 預算明細用條列格式逐項列出，禁止用空格或 Tab 對齊欄位（會形成偽表格）
  範例：・機票（台北-大阪來回）：約 30,000 日圓（約 6,000 台幣）
- 章節標題直接寫文字並換行，不要加任何符號裝飾
- 費用以實際匯率換算

輸出長度限制：
- 整體控制在 1000 字以內
- 偏好分析最多 3 段，每段「原文依據／推理結果／行程影響」合起來 1-2 句話即可
- 每日行程用條列式：每行格式「時間 景點（費用）交通」一行寫完，不要長篇說明
  範例：・09:00 大阪城天守閣（1,200日圓）地鐵谷町四丁目站步行10分
- 預算、住宿、交通、注意事項合計不超過 200 字
- 不要重複資訊（門票寫了景點段就別重複預算段細項）
- 禁止寫景點歷史背景、典故、文化說明，只列實用資訊"""

REFLECT_SYSTEM_PROMPT = """你是一位嚴謹的旅遊行程審核員。
請客觀評估行程草案的可行性，重點檢查以下項目：
1. 資訊溯源（最重要）：行程裡每個具體實體——飯店名稱、景點、門票/住宿價格、交通方式——
   都必須能在「Search 蒐集到的外部資訊」中找到對應。逐項核對，凡是 Search 資訊裡
   找不到的（例如冒出一家資訊清單上沒有的飯店、一個沒查過的價格），就是未經查證的內容。
2. 偏好符合度：對照「使用者偏好」資料，逐項檢查行程是否真的呼應其旅遊風格——
   住宿類型（如設計感、老屋改建）、餐飲取向（如在地小吃、高CP值）、景點偏好
   （如歷史建築、文化場館）、天氣應對習慣（如雨天排室內）。不符之處要明確指出。
3. 預算合理性：使用實際匯率計算，確認總花費是否在合理範圍內
4. 天氣適配性：是否考慮天氣影響，雨天是否有室內備案
5. 動線合理性：每日景點之間的交通動線是否順暢

評估結果（第一行只能是以下三者之一）：
- PASS — 行程通過評估，且所有具體實體都能在 Search 資訊中找到對應；住宿/餐飲估算價格若已清楚標註估算與以訂房網站或現場為準，也可通過。
- REVISE — 行程安排要改，或出現未經查證的內容但 Search 資訊裡有同類資料可替換
  （例如 generate 編了一家清單外的飯店，但清單其實有其他飯店可選）。
  接著逐項說明：哪些內容未經查證、應改用 Search 資訊中的哪一筆。
- RESEARCH — 外部資訊本身不足或有誤需要重查（例如 Search 資訊整類缺漏：完全沒有住宿資料、
  缺住宿價格、缺餐飲價格、缺某天天氣、匯率過期、門票數字可疑），接著明確指出「缺哪一類資訊」。

判斷原則：
- generate 寫了 Search 資訊裡沒有的東西，但同類資訊存在（有別的飯店/景點可用）→ REVISE，要求改用查過的資料。
- generate 把 Search 找不到的住宿/餐飲價格寫成確定價格，且沒有「估算」與「實際以訂房網站/現場為準」→ RESEARCH，要求補查價格。
- Search 已補查但仍明確寫「價格未查得可靠來源」，而 generate 已用「估算：約X日圓，實際以訂房網站/現場為準」標註 → 不得只因該估算價格判 REVISE。
- 若 generate 把周遊卡或套票已涵蓋門票又列入總預算，造成重複計費 → REVISE。
- 問題根源是「Search 資訊整類缺漏或明顯錯誤」→ RESEARCH，回去補查。
- 全部對得上、安排也合理 → PASS。

輸出格式要求：
- 全程使用純文字，禁止任何 markdown 語法
- 禁止使用：# ## ### 標題符號、**粗體**、*斜體*、`程式碼`、表格、---- 分隔線、> 引言、emoji
- 條列請用「・」或「1. 2. 3.」開頭，不要用「-」或「*」"""


def _build_state_summary(state: TravelState) -> str:
    """把目前 State 整理成 brain 看得懂的進度摘要。"""
    lines = []
    # next_action 是上一輪 brain 選的動作，亦即「剛剛執行完的動作」。
    last_action = state.get("next_action") or "（無，這是本輪第一次決策）"
    lines.append(f"・上一個動作：{last_action}")

    lines.append(f"・偏好檢索（retrieve）：{'已完成' if state.get('preferences') else '尚未進行'}")
    lines.append(f"・外部資訊（search）：{'已蒐集' if state.get('external_info') else '尚未蒐集'}")

    if state.get("draft_itinerary"):
        lines.append(
            f"・行程草案（generate）：已產出（已生成 {state.get('revision_count', 0)} 版）"
        )
    else:
        lines.append("・行程草案（generate）：尚未產出")

    if state.get("reflection"):
        # 直接取 reflection 第一行（PASS / REVISE / RESEARCH）當判定詞，
        # 讓 brain 能區分「改行程」還是「重查資訊」。
        verdict = state["reflection"].strip().split("\n")[0][:20]
        excerpt = state["reflection"][:200].replace("\n", " ")
        lines.append(f"・可行性評估（reflect）：最近結果 {verdict}")
        lines.append(f"  審核意見摘要：{excerpt}")
    else:
        lines.append("・可行性評估（reflect）：尚未評估")

    lines.append(f"・已執行 brain 決策次數：{state.get('step_count', 0)}")
    return "\n".join(lines)


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
    brain_llm = llm.with_structured_output(BrainDecision)

    # ──────────────────────────────────────────
    # 節點 0：Brain（決策中樞）
    # ──────────────────────────────────────────
    async def brain_node(state: TravelState) -> dict:
        user_query = _get_user_query(state["messages"])
        history = _format_history(state["messages"])
        summary = _build_state_summary(state)

        prompt = (
            f"[對話記錄]\n{history}\n\n"
            f"[本輪使用者需求]\n{user_query}\n\n"
            f"[目前進度]\n{summary}"
        )

        decision: BrainDecision = await brain_llm.ainvoke(
            [
                SystemMessage(content=BRAIN_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        return {
            "next_action": decision.action,
            "brain_reasoning": decision.reasoning,
            "step_count": state.get("step_count", 0) + 1,
        }

    # ──────────────────────────────────────────
    # 節點 1：Retrieve（偏好檢索）
    # ──────────────────────────────────────────
    async def retrieve_node(state: TravelState) -> dict:
        user_query = _get_user_query(state["messages"])
        query = build_preference_query(user_query)
        docs = retriever.invoke(query)
        preferences = "\n\n".join(doc.page_content for doc in docs)
        return {"preferences": preferences}

    # ──────────────────────────────────────────
    # 節點 2：Search（資訊蒐集）
    # ──────────────────────────────────────────
    async def search_node(state: TravelState) -> dict:
        llm_with_tools = llm.bind_tools(tools)
        today = datetime.now().strftime("%Y-%m-%d")

        user_query = _get_user_query(state["messages"])
        history = _format_history(state["messages"])

        # 若 reflect 判定資訊不足而觸發重查，把審核意見帶進來，引導針對性補查。
        research_hint = ""
        if state.get("reflection", "").strip().startswith("RESEARCH"):
            research_hint = (
                f"\n[!!!審核發現資訊不足，請針對以下缺口補查]\n{state['reflection']}\n"
                f"請優先補齊上述缺漏或可疑的資訊，不要只重複先前的搜尋。\n"
            )

        messages = [
            SystemMessage(content=SEARCH_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"今日日期：{today}\n"
                    f"[對話記錄]\n{history}\n\n"
                    f"[本次使用者旅遊需求]\n{user_query}\n\n"
                    f"[使用者偏好（來自過往旅遊紀錄，搜尋時請據此做針對性查詢）]\n"
                    f"{state['preferences']}\n"
                    f"{research_hint}"
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

                # MCP 工具偶爾會因網路或服務問題失敗，包 try/except 讓流程不中斷
                try:
                    result = await tool_map[tc["name"]].ainvoke(tc["args"])
                except Exception as e:
                    result = f"工具呼叫失敗：{e}"

                print(f"工具結果：{_preview_tool_value(result)}", flush=True)
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        external_info = response.content.strip() or "外部資訊查詢未取得結果"
        return {"external_info": external_info}

    # ──────────────────────────────────────────
    # 節點 3：Generate（行程生成）
    # ──────────────────────────────────────────
    async def generate_node(state: TravelState) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        user_query = _get_user_query(state["messages"])
        history = _format_history(state["messages"])

        # 如果是重新規劃，把 reflection 放在最前面、最顯眼
        if state["reflection"]:
            prompt = (
                f"[!!!重要：這是第 {state['revision_count']} 次修正，必須處理以下問題]\n"
                f"{state['reflection']}\n\n"
                f"修正規則：\n"
                f"・上述每一項問題都必須在新行程中明確處理（修正數字、改地點、補備案等）\n"
                f"・不要只是改寫文字而保留同樣數字或同樣安排\n"
                f"・行程正文寫完後，在最後另起一行「===修正摘要===」，逐項說明「問題X：原本→改成」。\n"
                f"  這段僅供內部審核，不會顯示給使用者，所以一定要放在最後面。\n\n"
                f"────────────────\n\n"
                f"[今日日期：{today}]\n\n"
                f"[使用者旅遊需求]\n{user_query}\n\n"
                f"[對話記錄]\n{history}\n\n"
                f"[使用者旅遊偏好]\n{state['preferences']}\n\n"
                f"[即時資訊]\n{state['external_info']}"
            )
        else:
            prompt = (
                f"[今日日期：{today}]\n\n"
                f"[使用者旅遊需求]\n{user_query}\n\n"
                f"[對話記錄]\n{history}\n\n"
                f"[使用者旅遊偏好（來自過往旅遊紀錄）]\n{state['preferences']}\n\n"
                f"[即時資訊（景點、天氣、匯率）]\n{state['external_info']}"
            )

        messages = [
            SystemMessage(content=GENERATE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        # 使用 streaming 避免 NVIDIA 網關 504 timeout（長輸出會被砍）
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)

        # revision_count 在 generate 自增，記錄已產出第幾版草案，供 brain 判斷是否收斂
        return {
            "draft_itinerary": "".join(chunks),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    # ──────────────────────────────────────────
    # 節點 4：Reflect（可行性評估）
    # ──────────────────────────────────────────
    async def reflect_node(state: TravelState) -> dict:
        user_query = _get_user_query(state["messages"])
        prompt = (
            f"[行程草案]\n{state['draft_itinerary']}\n\n"
            f"[使用者原始需求]\n{user_query}\n\n"
            f"[使用者偏好（來自過往旅遊紀錄）]\n{state['preferences']}\n\n"
            f"[Search 蒐集到的外部資訊（景點門票、天氣、匯率、交通、住宿）]\n"
            f"{state['external_info']}\n\n"
            f"請對照外部資訊檢查行程數字是否一致、有無缺漏；"
            f"並對照使用者偏好檢查行程是否真的呼應其旅遊風格。"
        )

        messages = [
            SystemMessage(content=REFLECT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)
        content = "".join(chunks)
        # 評估結果(PASS/REVISE/RESEARCH)寫在 reflection 第一行，由 brain 讀取分流。
        return {"reflection": content}

    # ──────────────────────────────────────────
    # 節點 5：Respond（回覆使用者）
    # ──────────────────────────────────────────
    async def respond_node(state: TravelState) -> dict:
        # 修正版草案會在結尾附「===修正摘要===」內部審核段落，
        # 切掉它，只把行程正文回覆給使用者。
        draft = state["draft_itinerary"].split("===修正摘要===")[0].rstrip()
        final = draft or "目前還沒有足夠資訊產出行程，請補充旅遊需求後再試一次。"
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
        }

    # 回傳所有節點
    return {
        "brain": brain_node,
        "retrieve": retrieve_node,
        "search": search_node,
        "generate": generate_node,
        "reflect": reflect_node,
        "respond": respond_node,
    }
