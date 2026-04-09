# Semantic Search CLI

Search your PDFs and notes with natural language — no keyword matching required.

Built from scratch as the retrieval core of a RAG pipeline, without relying on LangChain or a dedicated vector database.

## How It Works

```
[Index]   Document → chunk → embed (384-dim vector) → store in SQLite
[Search]  Query    → embed → cosine similarity against all chunks → return top-K
```

Semantically similar text produces similar vectors, so queries like *"CPU scheduling"* can match passages about *"process dispatch algorithms"* even with no keyword overlap.

## Project Structure

```
semantic-search/
├── main.py          # CLI entry point — index / search / list / clear
├── chunker.py       # File reader (PDF/TXT/MD) + sliding-window chunker
├── embedder.py      # sentence-transformers encoding + cosine similarity
├── store.py         # SQLite persistence for chunks and embedding blobs
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The model (`all-MiniLM-L6-v2`, ~90 MB) downloads on first run and is cached locally.

## Usage

```bash
# Index a folder or a single file
python main.py index ~/Desktop/notes/
python main.py index ~/Downloads/lecture.pdf

# Search with natural language
python main.py search "difference between processes and threads"
python main.py search "what is dynamic programming" --top 3

# List indexed files
python main.py list

# Clear the index
python main.py clear
```

## Design Decisions & Trade-offs

| Component | Choice | Alternative |
|-----------|--------|-------------|
| Chunking | Fixed-size sliding window (300 chars, 50 overlap) | Semantic chunking (higher quality, 2× slower indexing) |
| Embedding | `all-MiniLM-L6-v2` — fast, 22 MB, 384-dim | `paraphrase-multilingual-MiniLM-L12-v2` for better multilingual support |
| Vector store | SQLite with float32 BLOB serialization | Faiss / ChromaDB for million-scale corpora |
| Similarity | L2-normalised dot product (≡ cosine similarity) | Euclidean distance (sensitive to vector magnitude) |
| Retrieval | Single-stage top-K | Two-stage: coarse retrieval → cross-encoder rerank |

**Bottleneck:** `load_all()` reads the full embedding matrix into memory on every query. This is fine up to tens of thousands of chunks; beyond that, replace with a Faiss index and keep the matrix resident in memory.

## Relation to RAG

| This project | Full RAG |
|---|---|
| Chunking + embedding | ✅ identical |
| SQLite vector store | swap for ChromaDB / Faiss |
| Returns raw passages | feed passages into LLM prompt → generate answer |

To extend this into a full RAG system, pass the top-K retrieved passages as context to an LLM after `cmd_search`.

## Extending the Project

- `chunker.py` — tune `chunk_size` and `overlap` to improve retrieval precision
- `embedder.py` — swap the model; add a cross-encoder reranking pass after top-K retrieval
- `store.py` — replace SQLite with Faiss for large-scale corpora
- `main.py` — add an interactive REPL mode, file-type filters, or a confidence score display
