# ============================================================
# tools.py — executor 的工具組裝
# ============================================================
# 四個工具交給 create_agent 的 ReAct 迴圈自主調度：
#   retrieve_preferences — RAG over travel_docs（檢偏好）
#   tavily_search        — Tavily MCP（景點/簽證等即時資訊）
#   get_weather          — Open-Meteo MCP（天氣，調整室內外安排）
#   get_exchange_rate    — Frankfurter MCP（匯率，做預算可行性判斷）
# RAG 自己包成 @tool；其餘三類由 MCP server 提供，整批帶入。
# ============================================================

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag import PREFERENCE_QUERY

load_dotenv()


# ===================== MCP server 設定 =====================
def build_mcp_server_config() -> dict:
    return {
        "tavily": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            "transport": "stdio",
        },
        "open-meteo": {
            "command": "npx",
            "args": ["-y", "open-meteo-mcp-server"],
            "transport": "stdio",
        },
        "frankfurter": {
            "url": "https://mcp.frankfurter.dev/",
            "transport": "http",
        },
    }


# ===================== RAG → tool =====================
# 兩段檢索 union：
#   1) 固定 PREFERENCE_QUERY → 保證跨旅程的「通用好惡/心得」一定撈得到
#      （像「太觀光化」這種貫穿所有旅程的特質，不會被單一主題擠掉）。
#   2) LLM 把本次需求改寫成「去目的地、聚焦風格與好惡」的 query → 對齊這趟主題。
# 改寫只停在中立層（主題＋好惡），不點名任何結論，偏好仍由模型讀原文後自行歸納。
_REWRITE_PROMPT = """你是檢索查詢改寫器。請把使用者的旅遊需求，改寫成一段用來檢索「使用者過往旅行偏好」的查詢語句。
規則：
- 同時涵蓋兩塊：(a) 這趟旅程的主題與風格（如古蹟、文化、自然、放鬆、步調、住宿與餐飲取向）；
  (b) 使用者對景點/住宿/行程的好惡與評價（覺得值得、推薦的，以及失望、踩雷、想避開的）。
- 移除目的地的國名與城市名（如日本、大阪）——那些與過往國內紀錄語意不合，只會干擾檢索。
- 不要預設或臆測使用者的具體好惡結論，只描述「要檢索哪些面向」。
- 只輸出查詢語句本身，不要解釋。"""


def build_retrieve_tool(retriever, llm):
    def _dedup(docs):
        seen, uniq = set(), []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                uniq.append(d)
        return uniq

    @tool
    async def retrieve_preferences(query: str = "") -> str:
        """檢索使用者過往旅遊紀錄，推斷其旅行偏好（景點類型、住宿、預算分配、飲食與交通習慣、好惡與雷點）。規劃行程前先呼叫一次以對齊使用者風格。query 直接傳本次旅遊需求即可，工具會自行整理成偏好檢索語句。"""
        # 1) 用 LLM 把需求改寫成去目的地、聚焦風格與好惡的檢索語句
        themed_query = ""
        if query.strip():
            resp = await llm.ainvoke([
                SystemMessage(content=_REWRITE_PROMPT),
                HumanMessage(content=query),
            ])
            themed_query = resp.content.strip()
        # 2) 兩段檢索 union（固定偏好 query + 改寫後主題 query），去重後回傳
        docs = retriever.invoke(PREFERENCE_QUERY)
        if themed_query:
            docs += retriever.invoke(themed_query)
        return "\n\n".join(d.page_content for d in _dedup(docs))

    return retrieve_preferences


# ===================== 組裝完整工具清單 =====================
# 工具容錯：MCP 工具參數錯誤會丟 ToolException，預設會一路冒泡讓整支程式崩潰。
# 改以此 handler 把錯誤包成 ToolMessage 回饋給 LLM，附上錯誤原文與修正提示，讓它自行重試。
def _tool_error_hint(e: Exception) -> str:
    return (
        f"工具呼叫失敗：{e}\n"
        "請檢查並修正參數後重試（注意互斥參數不可同時帶入，例如 forecast_days "
        "與 start_date/end_date 只能擇一）。"
    )


async def build_tools(retriever, llm):
    client = MultiServerMCPClient(build_mcp_server_config())
    mcp_tools = await client.get_tools()
    tools = [build_retrieve_tool(retriever, llm), *mcp_tools]
    for t in tools:
        t.handle_tool_error = _tool_error_hint
    print(f"已組裝 {len(tools)} 個工具（1 個 RAG + {len(mcp_tools)} 個 MCP）")
    return tools
