"""
Vector RAG - 旅遊偏好檢索器
=========================
rag.py 負責將 ./data 中的旅遊紀錄轉成向量資料，並建立可供 main.py 查詢的 retriever。

程式流程：
  1. 從 data/ 讀取過往旅遊紀錄文字檔。
  2. 將 Documents 切成適合檢索的文字片段。
  3. 使用 NVIDIA NIM Embedding Model 將片段向量化。
  4. 在 Milvus 建立 travel_preferences collection。
  5. 將向量資料庫轉成 retriever，供偏好檢索節點使用。
"""

# ── 載入套件 ──────────────────────────────────────────────
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── 建立旅遊偏好向量索引與 Retriever ─────────────────────
def build_retriever():
    """讀取 ./data 旅遊紀錄、切片向量化存入 Milvus，回傳供節點查詢偏好的 retriever。"""
    # 先顯示目前正在重建向量資料，提醒使用者啟動階段可能需要等待。
    print("🔨 建立 Milvus collection，讀取 ./data")

    # DirectoryLoader 會找出 data/ 底下所有 .txt 旅遊紀錄。
    documents = DirectoryLoader(
        "./data",  # 旅遊紀錄所在的資料夾
        glob="**/*.txt",  # 包含子資料夾內的所有文字檔
        loader_cls=TextLoader,  # 每個檔案使用 TextLoader 讀成 Document
        loader_kwargs={"encoding": "utf-8"},  # 中文檔案固定使用 UTF-8 解碼
    ).load()

    # 將較長的旅行紀錄切成適合向量檢索的文字片段。
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=256,  # 每個片段最多約 256 個字元
        chunk_overlap=50,  # 相鄰片段重疊 50 個字元，保留跨段語意
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 優先沿中文句子邊界切分
    ).split_documents(documents)

    # 將所有片段交給 NVIDIA Embedding Model 向量化，再寫入 Milvus。
    vector_store = Milvus.from_documents(
        documents=chunks,  # 要建立索引的旅行紀錄片段
        embedding=NVIDIAEmbeddings(model=os.getenv("EMBEDDING_MODEL")),  # .env 指定的 Embedding Model
        collection_name="travel_preferences",  # 儲存旅遊偏好的 collection 名稱
        connection_args={"uri": "http://localhost:19530"},  # 本機 Milvus 服務位址
        drop_old=True,  # 每次啟動先刪除舊 collection，確保資料與 data/ 一致
    )
    # 顯示實際建立的向量筆數與儲存位置，方便確認索引完成。
    print(f"✅ 已建立 {len(chunks)} 筆向量，存入 http://localhost:19530")

    # 將向量資料庫轉成 Retriever；每次查詢回傳最相關的 5 個片段。
    return vector_store.as_retriever(search_kwargs={"k": 5})
