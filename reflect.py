# ============================================================
# reflect.py — reflect 節點（Reflection 核心）
# ============================================================
# 對 executor 產出的行程草案做多面向品質檢查，用 with_structured_output
# 強制回傳 Critique（verdict + issues），讓條件邊能可靠判讀，
# 不必去 parse 自由文字。回傳 critique 與 revisions+1。
# ============================================================

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from state import Critique, TravelState


# 單則工具結果的截斷長度：Tavily 結果可達上萬字，全帶會撐爆 context。
_TOOL_RESULT_CHARS = 1500


# ===================== 多面向審核指引 =====================
REFLECT_SYSTEM_PROMPT = """你是嚴謹的旅遊行程審核員。你自己沒有工具，但下方會附上 executor 本輪「實際用工具查回的原始資料」，
這就是你查核事實的唯一依據。請對行程草案做多面向品質檢查，逐項判斷是否有問題：

1. 偏好一致性：行程是否命中 RAG 推斷出的使用者偏好（住宿等級、預算分配、景點類型、交通習慣）。
2. 資訊時效/正確性（以下方工具結果為準，不要用你自己的記憶）：
   - 比對草案中的數字與事實（票價、開放時間、匯率、交通費、景點是否存在）是否與工具結果一致；
     不一致就標為問題，並指出「草案寫 X、工具結果是 Y」。
   - 草案出現工具結果裡完全沒有、也非常識的地點或數據 → 視為幻覺，標為問題。
   - 嚴禁用你自己的記憶去斷言「正確數值」。屬於可查證的固定事實（開放時間、是否休館、景點是否存在），
     工具沒涵蓋就標為「需 executor 用工具再確認 X」。
3. 行程合理性：地理動線、各景點間交通時間、開放時間與排程是否前後矛盾、天氣衝突是否處理（依工具回的降雨數據判斷室內外安排是否合理）。
4. 完整性：plan 各步驟是否執行完；住宿、交通、注意事項、預算明細是否齊全；跨國行程是否漏算機票等大項。
5. 預算可行性：以工具回的匯率換算後的總成本是否在使用者預算內、是否有漏項導致低估。

重要邊界——工具拿不到的東西不要當成 revise 理由：
機票票價、飯店房價、即時座位/空房這類「特定日期的即時報價與庫存」，現有工具（搜尋引擎）本來就查不到精確值，
executor 只能給合理估計。對這類項目，只把它寫成「提示：實際金額以訂票/訂房當下為準」放進 issues，
但「不可」因此判 revise（否則迴圈永遠無法收斂）。只有當草案與工具「明確查到」的數據相牴觸時，才構成 revise。

同理——查不到官方來源的固定事實（某景點門票），不要無限期硬擋：
若草案已誠實把某門票標為「以官網/現場為準」之類的提示（沒有謊稱官方、也沒有與工具明確查到的值牴觸），
代表 executor 查過但工具沒有可靠官方來源。此時當成「提示」即可，「不可」因「工具結果裡找不到這個數字」反覆判 revise
——否則同一個查不到的數字會每輪被打槍、永不收斂。只有當草案「謊稱官方」或與工具明確查到的值衝突時，才判 revise。

來源衝突的處理（很重要，避免反覆打槍同一項）：
同一個事實（例如周遊卡票價）工具結果可能有多筆、彼此數字不同（官網 vs 部落格 vs 比價站）。此時：
- 以「來源權威性」判斷，不要拿低權威來源去否定高權威來源：官方網站／官方票務頁 > 旅遊比價站 > 個人部落格/論壇。
- 「官方」的定義要嚴格：只有「景點/營運單位自己的官方網域」才算官方（例如 osakacastle.net、osaka-info.jp、tsutenkaku.co.jp、各景點官網）。
  FunTime、KKday、Klook、Trip.com、永安旅遊、雄獅、各旅遊部落格等都是「通路/比價站/部落格」，「一律不算官方」，
  不可把它們的數字當成「最高權威來源」去否定真正的官網。判斷某數字權威性時，先看它出自哪個網域。
- 若草案採用的數字與「最高權威來源（真官網）」一致，就算其他比價站/部落格寫不同數字，也「不可」判 revise，視為通過。
- 只有當草案的數字連真官網都對不上時，才判 revise，並指明「應以官方來源 X 的值為準」。
- 不要因為「某部落格/比價站寫的不一樣」就反覆要求修改——那會造成數字在兩個值之間來回震盪、永不收斂。
- 同一個官方頁面常列「多種票價」（大人/一般、大學生、高中生、兒童、敬老）。比對時一律以「大人/一般」票價為準，
  不要把學生票或優惠票誤當成大人價去打槍草案（例如官網大人 1,200、學生 600，草案寫 1,200 就是對的，別說「應為 600」）。

issues 清單只放「真正需要 executor 修改的問題」：
- 已確認正確、通過、無誤、合理的項目，「不要」列進 issues（不需要逐項回報你檢查過什麼，那是噪音，還會誤導判定）。
- 純屬「提示」性質（即時報價以訂票為準、門票以官網/現場為準等無法也不需修改的）也「不要」當成需修改的問題。
  若真的想保留提示，最多一條、並在句首標明「[提示]」。
- 每條 issue 一句話、果斷指出「哪個面向、哪裡要改、依據哪筆工具結果」，不要反覆糾結語意模糊的資料。

verdict 判定（務必與上面的 issues 一致）：
- 若 issues 裡「沒有任何一條是真正需要修改的問題」（全部都是確認正確、或頂多一條提示），verdict 必須是 "pass"。
- 只有當 issues 至少有一條明確指出「實際需要改動」時，才給 "revise"。
- 不要因為「你檢查了很多項」就給 revise；revise 的唯一理由是「還有東西要改」。"""


# ===================== 節點工廠 =====================
def create_reflect(llm):
    # function_calling 模式相容於 NVIDIA OpenAI 相容端點（既有 bind_tools 已驗證可用）
    critic = llm.with_structured_output(Critique, method="function_calling")

    async def reflect(state: TravelState) -> dict:
        # 取最後一則有內容的 AI 訊息，即 executor 產出的行程草案
        draft = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                draft = msg.content
                break

        user_query = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            "",
        )

        # 收集 executor 本輪實際用工具查回的原始資料，作為 reflect 查核事實的依據，
        # 讓它能比對「草案數字 vs 工具結果」而非憑記憶斷言。
        tool_results = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                name = getattr(msg, "name", None) or "tool"
                content = str(msg.content).strip()
                if len(content) > _TOOL_RESULT_CHARS:
                    content = f"{content[:_TOOL_RESULT_CHARS]}…（截斷）"
                tool_results.append(f"【{name}】{content}")
        tool_block = "\n\n".join(tool_results) if tool_results else "（本輪無工具結果）"

        critique = await critic.ainvoke([
            SystemMessage(content=REFLECT_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"[使用者需求與預算]\n{user_query}\n\n"
                f"[executor 實際用工具查回的原始資料]\n{tool_block}\n\n"
                f"[待審核的行程草案]\n{draft}"
            )),
        ])

        # 存進 state 前轉成純 dict：Critique 是 Pydantic 自訂型別，
        # 直接存進 checkpoint 會觸發 msgpack 未註冊型別警告（未來版本會被擋）。
        # Pydantic 只留在 with_structured_output 的輸出邊界做驗證。
        return {"critique": critique.model_dump(), "revisions": state.get("revisions", 0) + 1}

    return reflect
