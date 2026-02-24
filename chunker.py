"""
chunker.py — 读取文件并切成小块

为什么要切块：
  一篇文章很长，embedding 只能表达一小段文字的语义。
  切成 200~400 字的块，每块单独 embed，搜索更精准。
  这和 LangChain 的 RecursiveCharacterTextSplitter 是同一个思路。
"""

import pdfplumber
from pathlib import Path


def read_file(path: str | Path) -> str:
    """支持 .pdf / .txt / .md，返回纯文本"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"不支持的文件类型：{suffix}（支持 .pdf .txt .md）")


def split_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    按字符数切块，相邻块之间有 overlap 字符的重叠。
    重叠是为了避免答案恰好被切在两块的边界上。

    chunk_size=300  约等于一段话
    overlap=50      约等于一两句话的重叠
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # 下一块从 (start + chunk_size - overlap) 开始

    return chunks
