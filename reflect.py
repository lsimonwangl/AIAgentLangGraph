"""
Travel Agent - Reflect 節點（Reflection）
=======================================
reflect.py 負責對 executor 產出的行程草案做多面向品質檢查，
用 structured output 回傳 Critique（verdict + issues），讓條件邊能可靠判讀。

程式流程：
  1. 定義 Critique 結構，限制審核結果只能是 pass 或 revise。
  2. 建立結果審核提示詞，定義需求、偏好、完整性與合理性等檢查面向。
  3. 從 State 取出使用者需求、偏好資料與最新行程草案。
  4. 使用 LLM 產生結構化審核結果。
  5. 將 critique 與審核輪次寫回 State，交給條件邊判斷流程走向。
"""

# ── 載入套件 ──────────────────────────────────────────────
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import TravelState


# ── 定義 Reflect 的結構化輸出 ─────────────────────────────
class Critique(BaseModel):
    """reflect 節點的審核結果。

    用 llm.with_structured_output(Critique) 強制模型回傳固定結構，
    讓 graph 的條件邊能可靠判讀，不必解析自由文字。
    """

    # verdict 限制為兩種字串，讓條件邊可以直接判斷下一步。
    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    # issues 保存需要修改的具體問題；通過時使用空 list。
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")


# ── Reflect Prompt：結果審核面向與判定規則 ───────────────
def build_reflect_prompt() -> str:
    """建立審核提示詞，定義 reflect 的檢查面向與判定規則。"""
    return """\
你是旅遊行程結果審核員，請判斷目前行程是否已經足夠回答使用者需求。

請檢查：
1. 需求符合度：是否回應目的地、日期、天數、預算與其他明確條件。
2. 偏好一致性：是否符合偏好檔案中的景點、住宿、預算與交通習慣。
3. 完整性：是否包含每日行程、交通、住宿建議、預算與必要提醒。
4. 合理性：時間安排、地理動線、費用加總與前後內容是否有明顯矛盾。
5. 格式：是否符合每日行程與預算摘要的輸出要求。

若結果完整且沒有必須修改的問題，回傳 verdict="pass"，issues 留空。
只有當問題會影響主要行程，而且能靠修改目前草案解決時，才回傳 verdict="revise"。
issues 最多列 2 項，每項用一句話清楚指出要修改的內容。

不要自行查證或斷言即時票價、天氣、營業時間等外部資訊。
使用者未提供的出發機場、航班，以及即時房價、空房或取消條件，不可作為 revise 理由。
草案若已誠實標示費用為估算或待確認，也不可只因缺乏即時資料判 revise。
若只有提醒事項而沒有必要修改，必須判 pass。"""


# ── 建立 Reflect 節點 ─────────────────────────────────────
def create_reflect(llm):
    """建立 reflect 節點，回傳可註冊進 StateGraph 的 async 函式。"""
    # 套用 Critique 結構後，LLM 必須回傳 verdict 與 issues，不需再解析自由文字。
    # function_calling 模式相容於目前使用的 NVIDIA OpenAI 相容端點。
    critic = llm.with_structured_output(Critique, method="function_calling")

    # 內部 async 函式就是實際註冊到 StateGraph 的 Reflect 節點。
    async def reflect(state: TravelState) -> dict:
        # 先使用空字串初始化，避免找不到 AIMessage 時變數尚未定義。
        draft = ""
        # 對話歷史由新到舊搜尋，第一則有內容的 AIMessage 就是 Executor 最新草案。
        for msg in reversed(state["messages"]):
            # 略過 HumanMessage、ToolMessage 與沒有文字內容的 AIMessage。
            if isinstance(msg, AIMessage) and msg.content:
                draft = msg.content
                # 已取得最新草案，不需要繼續掃描更早的訊息。
                break

        # 同樣由新到舊取出最新 HumanMessage，作為需求符合度的審核基準。
        # 若歷史中沒有使用者訊息，next() 會回傳空字串。
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        # 將審核規則放在 SystemMessage，再用 HumanMessage 提供本輪實際審核資料。
        critique = await critic.ainvoke([
            SystemMessage(content=build_reflect_prompt()),  # 定義審核面向與 pass/revise 規則
            HumanMessage(content=(
                f"[使用者需求]\n{user_query}\n\n"  # 判斷草案是否回答本輪需求
                f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"  # 判斷個人化程度
                f"[待審核的行程草案]\n{draft}"  # Executor 最新產生的結果
            )),
        ])

        # Pydantic 物件直接進 checkpoint 會出現 msgpack 型別警告，因此先轉成一般 dict。
        # 每執行一次 Reflect 就將 revisions 加一，讓條件邊可以限制最大審核輪次。
        return {"critique": critique.model_dump(), "revisions": state.get("revisions", 0) + 1}

    # 回傳節點函式本身，main.py 之後會把它註冊到 StateGraph。
    return reflect
