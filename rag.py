"""
Vector RAG - 旅遊偏好檢索器
=========================
rag.py 負責將 ./data 中的旅遊紀錄轉成向量資料，並建立可供 main.py 查詢的 retriever。

執行流程：
    0. 載入套件與環境變數
    1. 從 ./data 讀取旅遊紀錄文字檔
    2. 將文字檔轉成 Document 資料結構
    3. 使用 RecursiveCharacterTextSplitter 將 Documents 切成 chunks
    4. 使用 NVIDIA NIM Embedding Model 將 chunks 向量化
    5. 透過 Milvus.from_documents 建立 travel_preferences collection
    6. 將向量資料庫轉成 retriever，供 main.py 查詢相關旅遊偏好

此模組提供 build_retriever() 函式供 main.py 呼叫。
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def build_preference_query(user_query: str) -> str:
    """回傳一段搜尋用文字，讓檢索器知道要找哪些旅行偏好資訊。"""
    return (
        f"根據使用者目前的旅遊需求「{user_query}」，"
        "從之前旅行紀錄中檢索與本次行程風格相關的內容，"
        "找出使用者的旅行風格偏好，包括景點類型、住宿評價、預算分配等個人化資訊，"
        "以及下次想去的地方。"
    )


def load_data_docs():
    """讀取 ./data 資料夾中的文字檔，轉成 LangChain Document 物件列表。"""
    loader = DirectoryLoader(
        "./data",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_docs(documents):
    """使用 RecursiveCharacterTextSplitter 將 documents 切成 chunks。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store():
    """建立 Milvus 向量資料庫，儲存可供語意搜尋的旅遊偏好資料。"""
    embeddings = NVIDIAEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
    )

    print("建立 Milvus collection，讀取 ./data")
    documents = load_data_docs()
    print(f"載入 {len(documents)} 份旅遊紀錄")

    chunks = split_docs(documents)
    print(f"切分為 {len(chunks)} 個 Chunks")

    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="travel_preferences",
        connection_args={"uri": "http://localhost:19530"},
        drop_old=True,
    )
    print(f"已建立 {len(chunks)} 筆向量，存入 http://localhost:19530")

    return vector_store


def build_retriever():
    """建立旅遊偏好檢索器，用於依照查詢文字取得相關的過往紀錄。"""
    vector_store = build_vector_store()

    return vector_store.as_retriever(search_kwargs={"k": 5})
