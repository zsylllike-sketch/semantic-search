"""
main.py — CLI 入口

用法：
  python main.py index  <文件或文件夹>   # 建索引
  python main.py search <问题>           # 搜索
  python main.py list                    # 查看已索引的文件
  python main.py clear                   # 清空索引

示例：
  python main.py index ~/Desktop/Courses/CS3305/
  python main.py index ~/Downloads/lecture.pdf
  python main.py search "进程和线程的区别"
  python main.py search "what is dynamic programming"
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

from chunker import read_file, split_chunks
from embedder import embed, cosine_similarity
from store import save_chunks, load_all, list_indexed, clear_db

console = Console()
SUPPORTED = {".pdf", ".txt", ".md"}


# ─────────────────────────── index ────────────────────────────

def cmd_index(paths: list[str]):
    """读取文件/文件夹，切块，生成 embedding，存入 SQLite"""
    files = collect_files(paths)
    if not files:
        rprint("[red]没有找到支持的文件（.pdf .txt .md）[/red]")
        sys.exit(1)

    console.print(f"\n[bold]📂 找到 {len(files)} 个文件，开始建索引...[/bold]\n")

    total_chunks = 0
    for file in track(files, description="处理中..."):
        try:
            text = read_file(file)
            chunks = split_chunks(text)
            if not chunks:
                console.print(f"  [yellow]⚠ 跳过空文件：{file.name}[/yellow]")
                continue
            embeddings = embed(chunks)
            save_chunks(str(file.name), chunks, embeddings)
            total_chunks += len(chunks)
            console.print(f"  [green]✓[/green] {file.name} → {len(chunks)} 块")
        except Exception as e:
            console.print(f"  [red]✗ {file.name}：{e}[/red]")

    console.print(f"\n[bold green]✅ 完成，共 {total_chunks} 个块已索引[/bold green]\n")


def collect_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            for f in path.rglob("*"):
                if f.suffix.lower() in SUPPORTED:
                    files.append(f)
        elif path.is_file() and path.suffix.lower() in SUPPORTED:
            files.append(path)
    return sorted(set(files))


# ─────────────────────────── search ───────────────────────────

def cmd_search(query: str, top_k: int = 5):
    """把问题 embed 后和库里所有 chunk 算相似度，取 top_k"""
    sources, texts, embeddings = load_all()

    if len(texts) == 0:
        rprint("[red]索引为空，请先运行 index 命令[/red]")
        sys.exit(1)

    # 把问题变成向量
    query_emb = embed([query])[0]

    # 和所有 chunk 算余弦相似度
    scores = cosine_similarity(query_emb, embeddings)

    # 取分数最高的 top_k
    top_indices = scores.argsort()[::-1][:top_k]

    console.print(f"\n[bold]🔍 查询：{query}[/bold]\n")

    for rank, idx in enumerate(top_indices, 1):
        score = scores[idx]
        if score < 0.2:  # 相似度太低就不展示
            break
        source = sources[idx]
        text   = texts[idx]

        # 截断显示，太长不好看
        display = text.replace("\n", " ").strip()
        if len(display) > 200:
            display = display[:200] + "..."

        console.print(f"[bold cyan]#{rank}[/bold cyan] [dim]{source}[/dim]  "
                      f"[yellow]相似度 {score:.3f}[/yellow]")
        console.print(f"  {display}\n")


# ─────────────────────────── list ─────────────────────────────

def cmd_list():
    rows = list_indexed()
    if not rows:
        rprint("[yellow]索引为空[/yellow]")
        return

    table = Table(title="已索引文件")
    table.add_column("文件名", style="cyan")
    table.add_column("块数", justify="right", style="green")
    for source, count in rows:
        table.add_row(source, str(count))
    console.print(table)


# ─────────────────────────── main ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="语义搜索 CLI — 用自然语言搜索你的文档",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py index ~/Desktop/Courses/CS3305/
  python main.py search "进程调度算法有哪些"
  python main.py search "dynamic programming" --top 3
  python main.py list
  python main.py clear
        """,
    )
    sub = parser.add_subparsers(dest="cmd")

    # index
    p_index = sub.add_parser("index", help="建索引：读取文件/文件夹")
    p_index.add_argument("paths", nargs="+", help="文件或文件夹路径")

    # search
    p_search = sub.add_parser("search", help="搜索")
    p_search.add_argument("query", help="搜索问题")
    p_search.add_argument("--top", type=int, default=5, help="返回条数（默认 5）")

    # list
    sub.add_parser("list", help="查看已索引的文件")

    # clear
    sub.add_parser("clear", help="清空索引数据库")

    args = parser.parse_args()

    if args.cmd == "index":
        cmd_index(args.paths)
    elif args.cmd == "search":
        cmd_search(args.query, args.top)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "clear":
        clear_db()
        rprint("[green]✅ 索引已清空[/green]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
