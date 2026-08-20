"""
Travel Agent - AI clients
=========================
clients.py 集中建立聊天模型與 Embedding client；MCP 工具仍由 tools.py 管理。
"""

import os

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI


DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:8317/v1"


def _required_env(name: str) -> str:
    """讀取必要環境變數，缺少時在啟動階段提供明確錯誤。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return value


def create_chat_model() -> ChatOpenAI:
    """建立供 planner、executor 與 reviewer 共用的本機 Proxy 聊天模型。"""
    return ChatOpenAI(
        base_url=os.getenv("CLI_PROXY_BASE_URL", DEFAULT_LOCAL_LLM_BASE_URL),
        api_key=os.getenv("CLI_PROXY_API_KEY", "123456"),
        model=_required_env("CHAT_MODEL"),
        reasoning_effort="low",
        timeout=60,
        max_retries=2,
    )


def create_embeddings() -> NVIDIAEmbeddings:
    """建立供旅遊偏好向量化使用的 NVIDIA Embedding client。"""
    return NVIDIAEmbeddings(model=_required_env("EMBEDDING_MODEL"))
