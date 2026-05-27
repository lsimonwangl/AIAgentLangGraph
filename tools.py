"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋與天氣查詢。

執行流程：
    0. 載入套件
    1. 建立 tavily MCP server 設定，並讀取 TAVILY_API_KEY
    2. 建立 open-meteo MCP server 設定，供 Agent 查詢天氣資訊
    3. 使用 MultiServerMCPClient 啟動 MCP tool servers
    4. 取得 LangChain Agent 可使用的 tools 清單
    5. 回傳 tools 清單給 main.py 使用

此模組提供 build_mcp_server_config()、load_mcp_tools() 與 get_all_tools() 函式供 main.py 呼叫。
"""

import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


def build_mcp_server_config() -> dict:
    """回傳 MCP 工具設定，告訴程式要啟動哪些外部工具服務。"""
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


async def get_all_tools():
    """啟動 MCP 工具服務，回傳完整工具清單。"""
    _client, tools = await load_mcp_tools()
    print(f"已載入 {len(tools)} 個 MCP 工具")
    return tools


async def load_mcp_tools():
    """依照設定啟動 MCP 工具，並回傳 MCP client 與工具清單。"""
    client = MultiServerMCPClient(build_mcp_server_config())
    tools = await client.get_tools()
    return client, tools
