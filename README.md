# Semantic Search CLI

用自然语言搜索你的课程 PDF / 笔记。

这是 RAG 的核心检索部分——不用 LangChain，从零理解原理。

## 项目结构

```
semantic-search/
├── main.py        ← CLI 入口，4 个命令：index / search / list / clear
├── chunker.py     ← 读取文件（PDF/TXT/MD），切成小块
├── embedder.py    ← 用 sentence-transformers 把文字变成向量
├── store.py       ← SQLite 存取 chunks 和 embeddings
└── requirements.txt
```

## 原理

```
[文件] → 切块 → embedding（向量） → 存入 SQLite
[问题] → embedding（向量） → 和库里所有向量算余弦相似度 → 取 top K
```

## 安装

```bash
pip install -r requirements.txt
```

首次运行会下载模型（~90MB），之后从缓存加载。

## 使用

```bash
# 建索引（支持文件夹递归）
python main.py index ~/Desktop/Courses/CS3305/
python main.py index ~/Downloads/lecture.pdf

# 搜索
python main.py search "进程和线程的区别"
python main.py search "what is dynamic programming" --top 3

# 查看已索引的文件
python main.py list

# 清空索引
python main.py clear
```

## 和 RAG 的关系

| 这个项目 | RAG（Phase 5）|
|---------|--------------|
| 切块 + embedding | ✅ 完全一样 |
| SQLite 存向量 | ChromaDB / FAISS 替代 |
| 返回原文 | 原文塞进 Prompt → LLM 生成回答 |

Phase 5 时只需要在 `cmd_search` 后面加一步：把检索到的文本塞进 LangChain 的 Prompt，就是完整的 RAG。

## 可以自己改的地方

- `chunker.py` → 调整 `chunk_size` 和 `overlap`，影响搜索精度
- `embedder.py` → 换其他模型（如 `paraphrase-multilingual-MiniLM-L12-v2` 中文效果更好）
- `main.py` → 加颜色高亮、加文件过滤、加交互模式
