"""
embedder.py — 把文字变成向量

sentence-transformers 会下载一个小模型（~90MB，首次慢，之后缓存）。
模型选 'all-MiniLM-L6-v2'：速度快、效果好、支持中英文混合。

embedding 是什么：
  一段文字 → 一个 384 维的 float 数组
  语义相近的文字，它们的向量在空间中距离也近
  这就是语义搜索的核心
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# 全局加载一次，避免重复初始化
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # 首次运行会下载模型，之后从本地缓存加载
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    把一批文本转成 embedding 矩阵
    返回 shape: (len(texts), 384)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # 归一化：让每个向量长度为 1，这样余弦相似度 = 点积，计算更快
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-9)


def cosine_similarity(query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
    """
    计算 query 和所有 corpus 向量的余弦相似度
    因为都已归一化，直接做点积即可
    返回 shape: (len(corpus),)，值域 [-1, 1]，越接近 1 越相似
    """
    return corpus_emb @ query_emb  # 矩阵乘法，你在 numpy 课里见过
