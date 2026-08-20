"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋、天氣查詢與匯率換算。

程式流程：
  1. 建立 Tavily、Open-Meteo 與 Frankfurter 三個 MCP Server 設定。
  2. 使用 MultiServerMCPClient 啟動外部工具服務。
  3. 為工具加入錯誤處理，讓 Agent 可以修正參數後重試。
  4. 將 MCP Client 與工具清單回傳給 main.py 使用。

偏好檢索不做成工具：那是每次規劃的必要輸入，由 retrieve_preferences 節點前置取得。

"""

# ── 載入套件 ──────────────────────────────────────────────
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


# ── 建立 MCP Server 設定 ──────────────────────────────────
def build_mcp_server_config() -> dict:
    """回傳 MCP (Model Context Protocol) 工具設定，告訴程式要啟動哪些外部工具服務。

    MCP 是一個讓 LLM 與外部工具溝通的標準協定。
    每個 server 設定包含以下欄位：
        - command:   啟動該工具服務的執行檔（這裡用 npx 直接執行 npm 套件）
        - args:      傳給 command 的參數；-y 表示自動同意安裝，package@version 固定版本
        - env:       要傳給該服務的環境變數（例如 API 金鑰）
        - transport: Agent 與 server 之間的溝通方式，stdio 代表透過標準輸入輸出，
                     http 代表連線到遠端 server
    """
    # 回傳的 dict 會直接交給 MultiServerMCPClient 建立三個工具服務。
    return {
        # tavily：提供網路搜尋功能，讓 Agent 能查詢景點/住宿/交通即時資訊
        "tavily": {
            "command": "npx",  # 使用 npx 啟動 npm 上的 MCP Server
            "args": ["-y", "tavily-mcp@0.2.22"],  # 固定 Tavily MCP 版本
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},  # 將搜尋金鑰傳給子程序
            "transport": "stdio",  # 透過標準輸入輸出與本機子程序通訊
        },
        # open-meteo：提供免費天氣查詢服務，不需 API 金鑰
        "open-meteo": {
            "command": "npx",  # 同樣使用 npx 啟動本機 MCP Server
            "args": ["-y", "open-meteo-mcp-server@2.0.1"],  # 固定天氣 MCP 版本
            "transport": "stdio",  # 透過標準輸入輸出通訊
        },
        # frankfurter：使用服務方管理的遠端 HTTP MCP，程式端沒有 npm 套件版本可固定
        "frankfurter": {
            "url": "https://mcp.frankfurter.dev/",  # 官方提供的遠端 MCP 位址
            "transport": "http",  # 直接透過 HTTP 連線，不啟動本機程序
        },
    }


# ── 將工具錯誤轉成 Agent 可讀的提示 ──────────────────────
def _tool_error_hint(error: Exception) -> str:
    """把工具錯誤轉成給 LLM 的修正提示，附上錯誤原文讓它自行重試。"""
    # 保留原始錯誤並補上常見參數問題，讓 Agent 有足夠資訊自行修正呼叫。
    return (
        f"工具呼叫失敗：{error}\n"
        "請檢查並修正參數後重試（注意互斥參數不可同時帶入，例如 forecast_days "
        "與 start_date/end_date 只能擇一）。"
    )


# ── 啟動 MCP 工具並加入錯誤處理 ───────────────────────────
async def load_mcp_tools():
    """依照設定啟動 MCP 工具，並回傳 Agent 可以直接使用的工具清單。"""
    # 讀取上方設定並建立 MCP Client；此時尚未真正取得工具清單。
    client = MultiServerMCPClient(build_mcp_server_config())

    # 連接所有 MCP Server，將它們提供的能力轉成 LangChain Tool 物件。
    tools = await client.get_tools()

    # 逐一替每個 Tool 設定相同的錯誤處理函式。
    for tool in tools:
        # 預設錯誤會中斷整張 Graph；改成 handler 後會把錯誤文字交回 Agent 判斷。
        tool.handle_tool_error = _tool_error_hint

    # 顯示實際取得的工具數量，方便確認三個 MCP Server 是否正常載入。
    print(f"✅ 已組裝 {len(tools)} 個 MCP 工具")
    # 同時回傳 client 與 tools；main.py 需保留 client，Executor 則只使用 tools。
    return client, tools
