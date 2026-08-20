"""
Travel Agent - Reflect 節點（Reflection）
=======================================
reflect.py 負責對 executor 產出的行程草案做多面向品質檢查，
用 structured output 回傳 Critique（verdict + issues），讓條件邊能可靠判讀。

執行流程：
    0. 載入套件
    1. 建立審核提示詞，定義需求、偏好、完整性與合理性等檢查面向
    2. 從對話歷史取出行程草案與使用者需求
    3. 將需求、偏好與草案餵給 LLM，產出結構化的審核結果
    4. 回傳 critique 與 revisions+1 寫入 state，交給條件邊路由

此模組提供 create_reflect() 函式供 main.py 呼叫。
"""

# 載入套件
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import TravelState


class Critique(BaseModel):
    """reflect 節點的審核結果。

    用 llm.with_structured_output(Critique) 強制模型回傳固定結構，
    讓 graph 的條件邊能可靠判讀，不必解析自由文字。
    """

    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")


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


def create_reflect(llm):
    """建立 reflect 節點，回傳可註冊進 StateGraph 的 async 函式。"""
    # function_calling 模式相容於 NVIDIA OpenAI 相容端點（既有 bind_tools 已驗證可用）
    critic = llm.with_structured_output(Critique, method="function_calling")

    async def reflect(state: TravelState) -> dict:
        # 取最後一則有內容的 AI 訊息，即 executor 產出的行程草案
        draft = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                draft = msg.content
                break

        # 取最新一則使用者訊息，作為需求符合度的審核基準
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        critique = await critic.ainvoke([
            SystemMessage(content=build_reflect_prompt()),
            HumanMessage(content=(
                f"[使用者需求]\n{user_query}\n\n"
                f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"
                f"[待審核的行程草案]\n{draft}"
            )),
        ])

        # 存進 state 前轉成純 dict：Pydantic 自訂型別直接存 checkpoint
        # 會觸發 msgpack 未註冊型別警告，只在 structured output 邊界用 Pydantic 驗證。
        return {"critique": critique.model_dump(), "revisions": state.get("revisions", 0) + 1}

    return reflect
