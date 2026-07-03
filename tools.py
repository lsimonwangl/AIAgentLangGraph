"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋、天氣查詢與匯率換算。

執行流程：
    0. 載入套件
    1. 建立 tavily MCP server 設定，並讀取 TAVILY_API_KEY
    2. 建立 open-meteo MCP server 設定，供 Agent 查詢天氣資訊
    3. 建立 frankfurter MCP server 設定，供 Agent 換算匯率
    4. 使用 MultiServerMCPClient 啟動 MCP tool servers
    5. 回傳 MCP client 與 tools 給 main.py 使用

偏好檢索不做成工具：那是每次規劃的必要輸入，由 retrieve_preferences 節點前置取得。

此模組提供 build_mcp_server_config() 與 load_mcp_tools() 函式供 main.py 呼叫。
"""

# 載入套件
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


def build_mcp_server_config() -> dict:
    """回傳 MCP (Model Context Protocol) 工具設定，告訴程式要啟動哪些外部工具服務。

    MCP 是一個讓 LLM 與外部工具溝通的標準協定。
    每個 server 設定包含以下欄位：
        - command:   啟動該工具服務的執行檔（這裡用 npx 直接執行 npm 套件）
        - args:      傳給 command 的參數；-y 表示自動同意安裝、@latest 抓最新版
        - env:       要傳給該服務的環境變數（例如 API 金鑰）
        - transport: Agent 與 server 之間的溝通方式，stdio 代表透過標準輸入輸出，
                     http 代表連線到遠端 server
    """
    return {
        # tavily：提供網路搜尋功能，讓 Agent 能查詢景點/住宿/交通即時資訊
        "tavily": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            "transport": "stdio",
        },
        # open-meteo：提供免費天氣查詢服務，不需 API 金鑰
        "open-meteo": {
            "command": "npx",
            "args": ["-y", "open-meteo-mcp-server"],
            "transport": "stdio",
        },
        # frankfurter：提供免費匯率換算服務，走遠端 HTTP server
        "frankfurter": {
            "url": "https://mcp.frankfurter.dev/",
            "transport": "http",
        },
    }


def _tool_error_hint(error: Exception) -> str:
    """把工具錯誤轉成給 LLM 的修正提示，附上錯誤原文讓它自行重試。"""
    return (
        f"工具呼叫失敗：{error}\n"
        "請檢查並修正參數後重試（注意互斥參數不可同時帶入，例如 forecast_days "
        "與 start_date/end_date 只能擇一）。"
    )


async def load_mcp_tools():
    """依照設定啟動 MCP 工具，並回傳 Agent 可以直接使用的工具清單。"""
    # 建立 MCP client，依照 server config 啟動外部工具服務
    client = MultiServerMCPClient(build_mcp_server_config())

    # 取得 Agent 可以直接使用的 tools 清單
    tools = await client.get_tools()

    # 工具錯誤容錯：handle_tool_error 預設 False，工具失敗會拋例外導致整個程式崩潰；
    # 改成 handler 後把錯誤包成訊息回饋 LLM，讓它換參數重試或略過，不中斷程式。
    for tool in tools:
        tool.handle_tool_error = _tool_error_hint

    print(f"✅ 已組裝 {len(tools)} 個 MCP 工具")
    return client, tools
