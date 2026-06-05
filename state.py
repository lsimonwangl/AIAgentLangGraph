# ============================================================
# state.py — TravelState 與 Critique 定義
# ============================================================
# 整張 graph 共享同一份 state。沿用官方主流做法：
#   - state 用 TypedDict 風格（MessagesState 底層即 TypedDict）
#   - Pydantic 只留在 critique 這個「LLM 輸出邊界」做驗證
# MessagesState 內建 messages: Annotated[list, add_messages]，
# 配合 checkpointer + thread_id 天然支援多 session 隔離與 interrupt 續跑。
# ============================================================

from typing import Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ===================== Critique：structured output =====================
# reflect 節點用 llm.with_structured_output(Critique) 強制模型回傳
# 可被條件邊可靠判讀的結構，避免去 parse 自由文字。
class Critique(BaseModel):
    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")


# ===================== TravelState：繼承 MessagesState =====================
# 只補三個額外欄位，其餘對話歷史交給內建的 messages channel。
class TravelState(MessagesState):
    plan: list[str]       # 可修改的計畫物件（Planning 核心）
    critique: dict | None # reflect 的結構化評估（Critique.model_dump()，存 dict 以利序列化）
    revisions: int        # 修正次數，設上限避免無窮迴圈
