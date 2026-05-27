"""
prompt.py — 讀取使用者輸入

對應 Lab1 Dify 的開始節點，負責從終端機接收使用者的旅遊需求。
"""


def read_query() -> str:
    user_input = input("\n請輸入旅遊需求（輸入 quit 結束）：")
    return user_input.strip()
