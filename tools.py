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
# 由 executor（ReAct agent）自主驅動檢索：它自己決定要查哪個偏好面向、不夠就換詞再查。
# 工具直接拿 agent 給的 query 去向量檢索（query 空才用 PREFERENCE_QUERY 當廣撈預設），
# 不做改寫、不做 union——「決定查什麼」的智能放在 agent 身上。
def build_retrieve_tool(retriever):
    @tool
    def retrieve_preferences(query: str = "") -> str:
        """檢索使用者過往旅遊紀錄中的偏好。一次查一個面向，可多次呼叫補齊。
        - query 請用「偏好面向」關鍵詞，例如：住宿價位與風格、飲食在地小吃或餐廳、
          行程步調、景點類型好惡、預算分配、踩雷與想避開的經驗。
        - 不要帶目的地國名/城市名（如大阪、日本），那與過往國內紀錄語意不合會干擾檢索。
        - 若規劃所需的某個偏好面向資訊還不足，請換關鍵詞「再呼叫一次」補齊。
        - query 留空則回傳一份綜合的偏好概覽，適合第一次廣撈。"""
        docs = retriever.invoke(query.strip() or PREFERENCE_QUERY)
        return "\n\n".join(d.page_content for d in docs)

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


async def build_tools(retriever):
    client = MultiServerMCPClient(build_mcp_server_config())
    mcp_tools = await client.get_tools()
    tools = [build_retrieve_tool(retriever), *mcp_tools]
    for t in tools:
        t.handle_tool_error = _tool_error_hint
    print(f"已組裝 {len(tools)} 個工具（1 個 RAG + {len(mcp_tools)} 個 MCP）")
    return tools
