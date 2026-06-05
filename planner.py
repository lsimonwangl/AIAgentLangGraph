# ============================================================
# planner.py — planner 節點（Planning）
# ============================================================
# 職責：
#   1. 讀使用者需求（與上一輪 critique，若有）產出/修訂可修改的 plan。
#   2. 收到 reflect 回拋的 critique 時，依 issues 自行診斷該怎麼修訂。
# plan 維持 list[str]（TypedDict 風格），不用 Pydantic 包整個 state。
# ============================================================

from datetime import date

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from state import TravelState


# ===================== 小工具 =====================
def _get_user_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _format_history(messages: list[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"使用者：{msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"助理：{msg.content}")
    return "\n".join(lines) if lines else "無"


# ===================== plan 產生 / 修訂 =====================
PLAN_SYSTEM_PROMPT = """你是旅遊規劃的任務分析師，負責把使用者的旅遊需求拆解成一份有序的執行計畫。
計畫每一步是一個可執行動作，後續會交給具備工具的 executor 依序執行。
標準步驟骨架（可依需求增減、調整順序）：
1. 用 RAG 檢索使用者過往旅遊偏好
2. 查目的地天氣以決定室內外安排
3. 查匯率以換算預算與成本
4. 搜尋景點、住宿、交通的即時資訊
5. 估算總成本並做預算可行性判斷
6. 生成每日行程草案

若收到上一輪的 critique，請針對 issues 逐項調整計畫（例如預算超支就加入「改用平價住宿/景點方案」步驟，動線不順就加入「重排地理動線」步驟），不要原封不動重列。

輸出規則：
- 每行一個步驟，用「1. 2. 3.」開頭
- 只輸出步驟清單，不要執行、不要解釋、不要 markdown 語法"""


def _parse_plan(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉行首的編號或項目符號
        for prefix in ("．", ".", "、", ")", "）"):
            if prefix in line[:4]:
                head, _, rest = line.partition(prefix)
                if head.strip().lstrip("0123456789") == "":
                    line = rest.strip()
                    break
        line = line.lstrip("・-*0123456789. 　").strip()
        if line:
            steps.append(line)
    return steps or [text.strip()]


# ===================== 節點工廠 =====================
def create_planner(llm):
    async def planner(state: TravelState) -> dict:
        user_query = _get_user_query(state["messages"])
        history = _format_history(state["messages"])

        # ── 帶入上一輪 critique（若有），讓 planner 自行診斷如何修訂 ──
        critique = state.get("critique")
        critique_block = "無（這是第一次規劃）"
        if critique is not None and critique.get("issues"):
            critique_block = "\n".join(f"・{i}" for i in critique["issues"])

        messages = [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"[今天日期] {date.today().isoformat()}（規劃涉及「下週二」等相對日期時以此為準）\n\n"
                f"[本次旅遊需求]\n{user_query}\n\n"
                f"[對話記錄]\n{history}\n\n"
                f"[上一輪審核發現的問題]\n{critique_block}"
            )),
        ]

        chunks = []
        async for chunk in llm.astream(messages):
            chunks.append(chunk.content)
        plan = _parse_plan("".join(chunks))

        return {"plan": plan}

    return planner
