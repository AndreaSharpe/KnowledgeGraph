#!/usr/bin/env python3
"""
用人工 gold 评估关系抽取（多分类，含 NA）。

用法：
  .venv\\Scripts\\python.exe scripts\\eval_re_gold.py --seed-type Person --split dev

默认读取：
- gold: data/gold/re_gold_{seed_type}_{split}.jsonl
- preds: data/curated/re_predictions_pcnn_{seed_type}.jsonl（或 merged 的 re_predictions.jsonl）

输出：
- data/analysis/re_eval_{seed_type}_{split}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.ds_labels import read_jsonl  # noqa: E402


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _eval(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    assert len(y_true) == len(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    per = {}
    for lb in labels:
        p = tp[lb] / (tp[lb] + fp[lb]) if (tp[lb] + fp[lb]) else 0.0
        r = tp[lb] / (tp[lb] + fn[lb]) if (tp[lb] + fn[lb]) else 0.0
        per[lb] = {"precision": p, "recall": r, "f1": _f1(p, r), "support": int(tp[lb] + fn[lb])}

    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0
    macro_f1 = sum(per[lb]["f1"] for lb in labels) / len(labels) if labels else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "labels": labels, "per_label": per}


def _apply_tau(pred_label: str, pred_prob: float, *, tau: float) -> str:
    if pred_label == "NA":
        return "NA"
    return pred_label if pred_prob >= tau else "NA"


def _best_tau(y_true: list[str], pred_label: list[str], pred_prob: list[float]) -> dict[str, Any]:
    # 只在非 NA 概率阈值上做简单网格搜索（经典且足够用）
    cands = sorted(set(float(x) for x in pred_prob))
    if not cands:
        return {"best_tau": 1.0, "best_macro_f1": 0.0}
    # 只取一些分位点避免过慢
    idxs = [int(len(cands) * q) for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]]
    grid = sorted({cands[min(max(i, 0), len(cands) - 1)] for i in idxs})

    best = {"best_tau": 1.0, "best_macro_f1": -1.0, "report": {}}
    for tau in grid:
        yp = [_apply_tau(l, p, tau=float(tau)) for l, p in zip(pred_label, pred_prob)]
        rep = _eval(y_true, yp)
        if rep["macro_f1"] > best["best_macro_f1"]:
            best = {"best_tau": float(tau), "best_macro_f1": float(rep["macro_f1"]), "report": rep}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="评估关系抽取 gold（多分类含 NA）")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    ap.add_argument("--tau", type=float, default=-1.0, help="非 NA 预测的最小概率阈值；<0 则自动搜索 best_tau")
    args = ap.parse_args()

    root = args.project_root.resolve()
    gold_path = root / "data" / "gold" / f"re_gold_{args.seed_type}_{args.split}.jsonl"
    if not gold_path.is_file():
        raise SystemExit(f"未找到 gold：{gold_path}（先运行 sample_re_gold.py 并补全 label）")

    gold_rows = read_jsonl(gold_path)
    gold = {str(r.get("bag_id") or ""): str(r.get("label") or "").strip() for r in gold_rows}
    gold = {k: v for k, v in gold.items() if k and v}
    if not gold:
        raise SystemExit(f"gold 文件里没有有效 label：{gold_path}")

    curated = root / "data" / "curated"
    p1 = curated / f"re_predictions_pcnn_{args.seed_type}.jsonl"
    p2 = curated / "re_predictions.jsonl"
    pred_rows = read_jsonl(p1 if p1.is_file() else p2)
    preds = {str(r.get("bag_id") or ""): r for r in pred_rows if str(r.get("seed_type") or "") == args.seed_type}

    y_true: list[str] = []
    pred_label: list[str] = []
    pred_prob: list[float] = []
    missing = 0
    for bid, yt in gold.items():
        pr = preds.get(bid)
        if not pr:
            missing += 1
            continue
        y_true.append(yt)
        pred_label.append(str(pr.get("predicted_label") or "NA"))
        pred_prob.append(float(pr.get("predicted_prob") or 0.0))

    if not y_true:
        raise SystemExit("没有可对齐的 gold/preds（bag_id 可能不一致）。")

    if args.tau >= 0:
        yp = [_apply_tau(l, p, tau=float(args.tau)) for l, p in zip(pred_label, pred_prob)]
        rep = _eval(y_true, yp)
        best = {"best_tau": float(args.tau), "best_macro_f1": float(rep["macro_f1"]), "report": rep}
    else:
        best = _best_tau(y_true, pred_label, pred_prob)

    out = {
        "seed_type": args.seed_type,
        "split": args.split,
        "n_gold": len(gold),
        "n_eval": len(y_true),
        "n_missing_preds": missing,
        "best_tau": best["best_tau"],
        "best_macro_f1": best["best_macro_f1"],
        "report_at_best_tau": best["report"],
    }

    out_dir = root / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"re_eval_{args.seed_type}_{args.split}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()

