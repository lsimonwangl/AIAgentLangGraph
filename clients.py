"""
Travel Agent - AI clients
=========================
clients.py 集中建立聊天模型與 Embedding client；MCP 工具仍由 tools.py 管理。
"""

import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _required_env(name: str) -> str:
    """讀取必要環境變數，缺少時在啟動階段提供明確錯誤。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return value


def create_chat_model() -> ChatOpenAI:
    """建立供 planner、executor 與 reflect 共用的 NVIDIA 聊天模型。"""
    return ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=_required_env("NVIDIA_API_KEY"),
        model=_required_env("CHAT_MODEL"),
        stream_chunk_timeout=300,
        max_retries=2,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=0.15,
            max_bucket_size=1,
        ),
    )


def create_embeddings() -> NVIDIAEmbeddings:
    """建立供旅遊偏好向量化使用的 NVIDIA Embedding client。"""
    return NVIDIAEmbeddings(model=_required_env("EMBEDDING_MODEL"))
