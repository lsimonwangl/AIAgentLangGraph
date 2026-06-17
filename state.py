"""
Travel Agent - State 定義
========================
state.py 負責定義整張 StateGraph 共享的 TravelState，以及 reflect 審核用的 Critique 結構。

執行流程：
    0. 載入套件
    1. 定義 Critique：reflect 節點 structured output 的驗證結構
    2. 定義 TravelState：繼承 MessagesState，補上偏好、計畫、審核結果與修正次數欄位

此模組提供 TravelState 與 Critique 供各節點模組使用。
"""

# 載入套件
from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class Critique(BaseModel):
    """reflect 節點的審核結果。

    用 llm.with_structured_output(Critique) 強制模型回傳固定結構，
    讓 graph 的條件邊能可靠判讀，不必解析自由文字。
    """

    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")


class TravelState(MessagesState):
    """整張 graph 共享的狀態。

    繼承 MessagesState 取得內建的 messages 欄位（多輪對話歷史），
    配合 checkpointer + thread_id 天然支援多 session 隔離。
    """

    preferences: str       # profile 節點檢索的偏好原文片段，planner/executor/reflect 共用同一份
    plan: list[str]        # planner 產出的可修改計畫（Planning 核心）
    critique: dict | None  # reflect 的審核結果（Critique.model_dump()，存 dict 以利 checkpoint 序列化）
    revisions: int         # 修正次數，達上限就結束迴圈避免無窮修訂
