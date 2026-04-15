#!/usr/bin/env python3
"""PCNN 推理：写入 data/curated/re_predictions*.jsonl（请使用 .venv 中的 python）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print("未安装 torch。请执行： .venv\\Scripts\\pip.exe install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from e

    from turing_kg.relation.pcnn_infer import infer_all_available, infer_and_write

    ap = argparse.ArgumentParser(description="PCNN 关系推理")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument(
        "--seed-type",
        type=str,
        default=None,
        choices=["Person", "Concept", "Award"],
        help="指定则只推理该类型并写入 re_predictions_pcnn_{类型}.jsonl；省略则合并写入 re_predictions.jsonl",
    )
    ap.add_argument("--ckpt", type=Path, default=None, help="单类型推理时可选自定义 checkpoint")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()
    root = args.project_root.resolve()
    dev = args.device or None

    if args.seed_type is not None:
        path, n = infer_and_write(root, seed_type=args.seed_type, ckpt_path=args.ckpt, device=dev)
        print(f"已写入 {path}，共 {n} 条 bag。")
    else:
        path, n = infer_all_available(root, device=dev)
        print(f"已合并写入 {path}，共 {n} 条 bag。")


if __name__ == "__main__":
    main()
