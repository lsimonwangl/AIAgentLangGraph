"""
Travel Agent - Planner 節點（Planning）
=====================================
planner.py 負責把使用者的旅遊需求拆解成有序的執行計畫，並依 reflect 的 critique 修訂。

執行流程：
    0. 載入套件
    1. 定義 Plan：structured output 的計畫結構（官方 plan-and-execute 慣用法，不做字串解析）
    2. 建立規劃提示詞，定義步驟骨架與偏好反映規則
    3. 將對話歷史、偏好檔案與上一輪 critique 餵給 LLM 產出計畫
    4. 回傳 plan 寫入 state，交給 executor 執行

此模組提供 create_planner() 函式供 main.py 呼叫。
"""

# 載入套件
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import TravelState


class Plan(BaseModel):
    """planner 的計畫結構，用 structured output 直接取得步驟清單。"""

    steps: list[str] = Field(description="有序的執行計畫，每個元素是一個可執行步驟")


def build_plan_prompt() -> str:
    """建立規劃提示詞，告訴 planner 怎麼拆解需求、反映偏好與處理 critique。"""
    return """\
你是旅遊規劃的任務分析師，負責把使用者的旅遊需求拆解成一份有序的執行計畫。
計畫每一步是一個可執行動作，後續會交給具備工具（搜尋/天氣/匯率）的 executor 依序執行。
標準步驟骨架（可依需求增減、調整順序）：
1. 查目的地天氣以決定室內外安排
2. 查匯率以換算預算與成本
3. 依偏好檔案搜尋景點、住宿、交通的即時資訊
4. 估算總成本並做預算可行性判斷
5. 生成每日行程草案

計畫必須反映偏好檔案：把「優先安排的類型」與「必須避開的類型」直接寫進搜尋與生成步驟
（例如偏好在地文創、不愛過度觀光化商圈，步驟就寫「搜尋在地文創景點與巷弄美食，避開純觀光商圈」），
不要寫「熱門景點」「必去景點」這類與偏好無關的通用步驟。

若收到上一輪的 critique，請針對 issues 逐項調整計畫（例如預算超支就加入「改用平價住宿/景點方案」步驟，動線不順就加入「重排地理動線」步驟），不要原封不動重列。

輸出規則：
- steps 的每個元素就是一個步驟的完整描述，不要加編號前綴
- 只產出計畫，不要執行步驟內容、不要解釋"""


def create_planner(llm):
    """建立 planner 節點，回傳可註冊進 StateGraph 的 async 函式。"""
    # function_calling 模式相容於 NVIDIA OpenAI 相容端點（與 reflect 相同作法）
    plan_llm = llm.with_structured_output(Plan, method="function_calling")

    async def planner(state: TravelState) -> dict:
        # 帶入上一輪 critique（若有），讓 planner 自行診斷如何修訂計畫
        critique = state.get("critique")
        critique_block = "無（這是第一次規劃）"
        if critique is not None and critique.get("issues"):
            critique_block = "\n".join(f"・{issue}" for issue in critique["issues"])

        # 本輪規劃指令：日期、偏好檔案與審核問題；對話歷史原樣以 message list 餵給模型
        directive = HumanMessage(content=(
            f"[今天日期] {date.today().isoformat()}（規劃涉及「下週二」等相對日期時以此為準）\n\n"
            f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"
            f"[上一輪審核發現的問題]\n{critique_block}\n\n"
            "請針對上方對話中最新一則旅遊需求，產出本輪執行計畫。"
        ))
        plan = await plan_llm.ainvoke([
            SystemMessage(content=build_plan_prompt()),
            *state["messages"],
            directive,
        ])
        return {"plan": plan.steps}

    return planner
