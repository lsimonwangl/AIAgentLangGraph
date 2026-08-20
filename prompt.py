"""
Travel Agent - 使用者輸入
=======================
prompt.py 負責從終端機讀取使用者每一輪輸入，並判斷是否要結束對話。

程式流程：
  1. 從終端機讀取使用者本輪輸入。
  2. 處理 Ctrl+C、Ctrl+D 與結束關鍵字。
  3. 將使用者問題、空字串或 None 回傳給對話迴圈。
"""


# ── 讀取使用者輸入 ────────────────────────────────────────
def read_query(turn: int) -> str | None:
    """從終端機讀取一輪使用者輸入，並回傳給對話迴圈判斷下一步。"""
    # input() 可能因使用者主動中斷而拋出例外，因此放在 try 區塊處理。
    try:
        # 顯示目前輪次並讀取輸入；strip() 會去掉前後多餘空白。
        query = input(f"[第 {turn} 輪] 你：").strip()
    except (EOFError, KeyboardInterrupt):
        # Ctrl+D 會產生 EOFError、Ctrl+C 會產生 KeyboardInterrupt，兩者都視為結束對話。
        return None

    # 將輸入轉成小寫後判斷，讓 exit、EXIT、Quit 等大小寫形式都能結束。
    if query.lower() in {"exit", "quit"}:
        # 回傳 None 通知 chat.py 跳出 while 迴圈。
        return None

    # 一般輸入直接回傳；若只按 Enter，會回傳空字串讓 chat.py 略過本輪。
    return query
