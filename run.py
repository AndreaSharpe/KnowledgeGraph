#!/usr/bin/env python3
"""唯一入口：构建图灵知识图谱并导出到 data/。

用法：
  python run.py
  python run.py --mode full
  python run.py --mode export_only
  python run.py --mode from_curated

模式说明见 sources/build_config.json（命令行 --mode 会覆盖配置文件）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from turing_kg.build import build_knowledge_graph, export_all


def main() -> None:
    ap = argparse.ArgumentParser(description="构建知识图谱并导出到 data/")
    ap.add_argument(
        "--mode",
        choices=["full", "export_only", "from_curated"],
        default=None,
        help="覆盖 sources/build_config.json 中的 mode；省略则读配置文件",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    g, triple_rows = build_knowledge_graph(root, mode=args.mode)
    export_all(root, g, triple_rows)
    print(f"完成。节点 {len(g.nodes)}，边 {len(g.edges)}。输出：{root / 'data'}")


if __name__ == "__main__":
    main()
