#!/usr/bin/env python3
"""从 sentences / routing / resolved 生成 data/curated/bags.jsonl（关系抽取 MIL 第一步）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.bags import build_and_write_bags  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 relation MIL 所需的 bags.jsonl")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="项目根目录（默认：脚本上级）",
    )
    ap.add_argument(
        "--min-zh-ratio",
        type=float,
        default=0.3,
        help="中文句子门控：zh_ratio >= 该值（默认 0.3）",
    )
    args = ap.parse_args()
    root = args.project_root.resolve()
    path, n = build_and_write_bags(root, min_zh_ratio=args.min_zh_ratio)
    print(f"已写入 {path}，共 {n} 个 bag。")


if __name__ == "__main__":
    main()
