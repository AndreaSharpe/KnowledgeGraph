#!/usr/bin/env python3
"""根据 bags.jsonl + Wikidata 声明生成 data/curated/ds_labels.jsonl（远程监督标签）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.ds_labels import build_and_write_ds_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 ds_labels.jsonl（需先存在 bags.jsonl）")
    ap.add_argument("--project-root", type=Path, default=ROOT, help="项目根目录")
    args = ap.parse_args()
    root = args.project_root.resolve()
    path, n = build_and_write_ds_labels(root)
    print(f"已写入 {path}，共 {n} 条。")


if __name__ == "__main__":
    main()
