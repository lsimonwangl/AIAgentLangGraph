"""
Travel Agent - Reviewer 節點（結果審查）
=======================================
reviewer.py 負責判斷 executor 產出的行程是否已有足夠資訊回答需求，
用 structured output 回傳 ReviewResult（verdict + issues），讓條件邊能可靠判讀。

執行流程：
    0. 載入套件
    1. 從對話歷史取出行程草案與使用者需求
    2. 對照使用者偏好，檢查需求符合度、資訊充分性、完整性與明顯矛盾
    3. 回傳 review 並累加 review_count，交給條件邊路由

此模組提供 create_reviewer() 函式供 main.py 呼叫。
"""

# 載入套件
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..state import TravelState


class ReviewResult(BaseModel):
    """reviewer 節點的結構化審查結果。

    用 llm.with_structured_output(ReviewResult) 強制模型回傳固定結構，
    讓 graph 的條件邊能可靠判讀，不必解析自由文字。
    """

    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")


REVIEWER_SYSTEM_PROMPT = """\
你是旅遊行程結果審查員，請判斷目前行程是否已有足夠資訊回答使用者需求。

請檢查：
1. 是否回應使用者的主要旅遊需求。
2. 是否符合使用者偏好。
3. 是否包含需求所需的日期、每日行程、交通、住宿建議與預算估算。
4. 時間與景點安排是否有明顯矛盾。
5. 是否還缺少必須透過工具補查的重要資訊。

如果資訊足夠且結果完整，回傳 verdict="pass"，issues 留空。
如果資訊不足或結果需要修改，回傳 verdict="revise"，並在 issues 中簡短說明缺少什麼或應修改什麼。
需要補充外部資訊時，issue 必須使用「需要補查：...」的格式，讓 planner 能將它轉成工具查詢步驟。
只有「現有工具與目前輸入確實能解決」且會影響行程主要內容的問題，才可判 revise。
下列情況不可作為 revise 理由：使用者未提供的出發機場或航班、即時訂房庫存與取消條件、
尚未進入預報範圍的天氣，以及草案已誠實標示為估算或待確認的資訊。
相對日期應依下方提供的今天日期判斷；日期推算正確時不可要求使用者再次確認。
不要自行查證或斷言即時票價、天氣、營業時間等外部資訊，也不要提出使用者沒有要求的額外條件。
issues 最多列 2 項；若只有提醒事項而沒有可執行的必要修改，必須判 pass。"""


def create_reviewer(llm):
    """建立 reviewer 節點，回傳可註冊進 StateGraph 的 async 函式。"""
    # function_calling 模式相容於 NVIDIA OpenAI 相容端點（既有 bind_tools 已驗證可用）
    reviewer_llm = llm.with_structured_output(ReviewResult, method="function_calling")

    async def reviewer(state: TravelState) -> dict:
        # 取最後一則有內容的 AI 訊息，即 executor 產出的行程草案
        draft = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                draft = msg.content
                break

        # 取最新一則使用者訊息，作為需求符合度的審查基準
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        review = await reviewer_llm.ainvoke([
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"[今天日期]\n{date.today().isoformat()}\n\n"
                f"[使用者需求]\n{user_query}\n\n"
                f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"
                f"[待審查的行程草案]\n{draft}"
            )),
        ])

        return {"review": review.model_dump(), "review_count": state.get("review_count", 0) + 1}

    return reviewer
