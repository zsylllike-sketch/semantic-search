"""
store.py — SQLite 存取 chunks 和 embeddings

你已经会 SQLite，这里只是把 embedding（一个 numpy 数组）
序列化成 bytes 存进去，取出来再反序列化。
"""

import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "index.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT NOT NULL,       -- 来源文件名
            chunk_id  INTEGER NOT NULL,    -- 第几块
            text      TEXT NOT NULL,       -- 原文
            embedding BLOB NOT NULL        -- numpy float32 数组序列化
        )
    """)
    conn.commit()
    return conn


def save_chunks(source: str, chunks: list[str], embeddings: np.ndarray):
    """把一个文件的所有 chunks + embeddings 存入数据库"""
    conn = get_conn()
    # 先删掉同一文件的旧数据（重新 index 时用）
    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    rows = [
        (source, i, text, emb.astype(np.float32).tobytes())
        for i, (text, emb) in enumerate(zip(chunks, embeddings))
    ]
    conn.executemany(
        "INSERT INTO chunks (source, chunk_id, text, embedding) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_all() -> tuple[list[str], list[str], np.ndarray]:
    """
    读出全部数据，返回 (sources, texts, embeddings_matrix)
    embeddings_matrix shape: (N, embedding_dim)
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT source, text, embedding FROM chunks ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        return [], [], np.empty((0, 0))

    sources = [r[0] for r in rows]
    texts   = [r[1] for r in rows]
    embeds  = np.stack([
        np.frombuffer(r[2], dtype=np.float32) for r in rows
    ])
    return sources, texts, embeds


def list_indexed() -> list[tuple[str, int]]:
    """列出已 index 的文件和各自的 chunk 数"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
    ).fetchall()
    conn.close()
    return rows


def clear_db():
    conn = get_conn()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()
