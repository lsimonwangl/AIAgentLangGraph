"""
Travel Agent - Planner 節點（Planning）
=====================================
planner.py 負責把使用者的旅遊需求拆解成有序的執行計畫，並依 reflect 的 critique 修訂。

程式流程：
  1. 定義 Plan 結構，讓 LLM 以 structured output 回傳步驟清單。
  2. 建立規劃提示詞，定義拆解需求與套用偏好的方式。
  3. 將對話歷史、偏好資料與上一輪 critique 交給 LLM。
  4. 將產生的 plan 寫入 State，交給 Executor 執行。
"""

# ── 載入套件 ──────────────────────────────────────────────
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import TravelState


# ── 定義 Planner 的結構化輸出 ─────────────────────────────
class Plan(BaseModel):
    """planner 的計畫結構，用 structured output 直接取得步驟清單。"""

    # 每個 list 元素都是一個可直接交給 Executor 執行的完整步驟。
    steps: list[str] = Field(description="有序的執行計畫，每個元素是一個可執行步驟")


# ── Planner Prompt：規劃步驟、偏好與修訂規則 ─────────────
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


# ── 建立 Planner 節點 ─────────────────────────────────────
def create_planner(llm):
    """建立 planner 節點，回傳可註冊進 StateGraph 的 async 函式。"""
    # 套用 Plan 結構後，LLM 必須透過 function calling 回傳固定的 steps 欄位。
    # function_calling 模式相容於目前使用的 NVIDIA OpenAI 相容端點。
    plan_llm = llm.with_structured_output(Plan, method="function_calling")

    # 內部 async 函式就是實際註冊到 StateGraph 的 Planner 節點。
    async def planner(state: TravelState) -> dict:
        # 從 State 讀取上一輪審核結果；第一次規劃時 critique 會是 None。
        critique = state.get("critique")
        # critique 為 None 時先轉成空 dict，避免直接呼叫 .get() 產生錯誤。
        issues = (critique or {}).get("issues")
        # issues 有內容代表本輪是修訂，沒有內容則代表第一次規劃。
        is_revision = bool(issues)
        # 將多個審核問題整理成 Prompt 可讀的條列文字。
        critique_block = "\n".join(f"・{i}" for i in issues) if is_revision else "無（這是第一次規劃）"

        # 修訂輪只要求修改有問題的部分，避免重新執行已完成且沒有問題的步驟。
        # 第一次規劃則要求依最新旅遊需求產生完整計畫。
        task = (
            "請只針對上方審核問題，產出「修正步驟」清單：每步對應一個 issue 寫「改什麼」。"
            "天氣、匯率、以及上一版已完成的景點/住宿/交通蒐集都不要重列——"
            "executor 會保留上一版草案，只改你列出的地方。"
            if is_revision else
            "請針對上方對話中最新一則旅遊需求，產出本輪執行計畫。"
        )

        # 組裝本輪一次性指令，加入日期、偏好資料、審核問題與本輪任務。
        directive = HumanMessage(content=(
            f"[今天日期] {date.today().isoformat()}（規劃涉及「下週二」等相對日期時以此為準）\n\n"
            f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"
            f"[上一輪審核發現的問題]\n{critique_block}\n\n"
            + task
        ))

        # 依序傳入系統規則、完整對話歷史與本輪指令，讓 LLM 產生結構化計畫。
        plan = await plan_llm.ainvoke([
            SystemMessage(content=build_plan_prompt()),  # 系統提示詞，定義規劃規則與步驟骨架
            *state["messages"],                          # 對話歷史，含使用者最新旅遊需求
            directive,                                   # 本輪規劃指令：日期、偏好檔案與審核問題
        ])
        # 只把 Plan 中的 steps 寫回 State，供下一個 Executor 節點讀取。
        return {"plan": plan.steps}

    # 回傳節點函式本身，main.py 之後會把它註冊到 StateGraph。
    return planner
