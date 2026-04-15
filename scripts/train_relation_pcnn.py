#!/usr/bin/env python3
"""
在 .venv 中运行（项目根目录）：
  .venv\\Scripts\\python.exe scripts\\train_relation_pcnn.py --seed-type Person

依赖：torch（见 requirements.txt）
"""

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
        print("未安装 torch。请在项目根目录执行： .venv\\Scripts\\pip.exe install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from e

    from turing_kg.relation.pcnn_train import train_pcnn_mil

    ap = argparse.ArgumentParser(description="PCNN + MIL-Attention 关系抽取训练")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--att-dim", type=int, default=128, help="MIL-Attention 中间维（Lin et al. 风格 u^T tanh(W h)）")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu，默认自动")
    ap.add_argument("--out-dir", type=Path, default=None, help="默认 models/relation_pcnn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = args.out_dir.resolve() if args.out_dir else None
    info = train_pcnn_mil(
        root,
        seed_type=args.seed_type,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        att_dim=args.att_dim,
        device=args.device or None,
        out_dir=out,
        seed=args.seed,
    )
    print(info)


if __name__ == "__main__":
    main()
