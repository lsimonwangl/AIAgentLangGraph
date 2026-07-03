"""
Vector RAG - 旅遊偏好檢索器
=========================
rag.py 負責將 ./data 中的旅遊紀錄轉成向量資料，並建立可供 main.py 查詢的 retriever。

執行流程：
    0. 載入套件
    1. 從 ./data 讀取旅遊紀錄文字檔
    2. 將文字檔轉成 Document 資料結構
    3. 使用 RecursiveCharacterTextSplitter 將 Documents 切成 chunks
    4. 使用 NVIDIA NIM Embedding Model 將 chunks 向量化
    5. 透過 Milvus.from_documents 建立 travel_preferences collection
    6. 將向量資料庫轉成 retriever，供 retrieve_preferences 節點查詢相關旅遊偏好

此模組提供 build_retriever() 函式供 main.py 呼叫。
"""

# 載入套件
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_retriever():
    """讀取 ./data 旅遊紀錄、切片向量化存入 Milvus，回傳供節點查詢偏好的 retriever。"""
    print("🔨 建立 Milvus collection，讀取 ./data")

    # 讀取 ./data 的 .txt 旅遊紀錄
    documents = DirectoryLoader(
        "./data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    # 切成可檢索片段，separators 保留中文語意邊界（句號/逗號），重疊保留部分上下文
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    ).split_documents(documents)

    # 用 NVIDIA NIM Embedding 向量化後建立 collection（drop_old：每次啟動重建）
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=NVIDIAEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
        collection_name="travel_preferences",
        connection_args={"uri": "http://localhost:19530"},
        drop_old=True,
    )
    print(f"✅ 已建立 {len(chunks)} 筆向量，存入 http://localhost:19530")

    return vector_store.as_retriever(search_kwargs={"k": 5})
