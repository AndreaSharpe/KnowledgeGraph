#!/usr/bin/env python3
"""
基于 gold(dev) 自动选择 best_tau，并写回 sources/relation_thresholds.json（by_seed_type）。

经典闭环：
1) sample_re_gold.py 抽样并人工标注（dev）
2) 本脚本在 dev 上搜索 best_tau（最大化 macro-F1）
3) 写入阈值配置；随后 export/ingest 时自动采用新的阈值
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.ds_labels import read_jsonl  # noqa: E402


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _eval_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    # macro-F1 over labels present in union (含 NA)
    labels = sorted(set(y_true) | set(y_pred))
    tp = {lb: 0 for lb in labels}
    fp = {lb: 0 for lb in labels}
    fn = {lb: 0 for lb in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] = fp.get(yp, 0) + 1
            fn[yt] = fn.get(yt, 0) + 1
    f1s = []
    for lb in labels:
        p = tp.get(lb, 0) / (tp.get(lb, 0) + fp.get(lb, 0)) if (tp.get(lb, 0) + fp.get(lb, 0)) else 0.0
        r = tp.get(lb, 0) / (tp.get(lb, 0) + fn.get(lb, 0)) if (tp.get(lb, 0) + fn.get(lb, 0)) else 0.0
        f1s.append(_f1(p, r))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _apply_tau(pred_label: str, pred_prob: float, *, tau: float) -> str:
    if pred_label == "NA":
        return "NA"
    return pred_label if pred_prob >= tau else "NA"


def _pick_best_tau(y_true: list[str], pred_label: list[str], pred_prob: list[float]) -> tuple[float, float]:
    cands = sorted(set(float(x) for x in pred_prob))
    if not cands:
        return 1.0, 0.0
    idxs = [int(len(cands) * q) for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]]
    grid = sorted({cands[min(max(i, 0), len(cands) - 1)] for i in idxs})
    best_tau = 1.0
    best_f1 = -1.0
    for tau in grid:
        yp = [_apply_tau(l, p, tau=float(tau)) for l, p in zip(pred_label, pred_prob)]
        mf1 = _eval_macro_f1(y_true, yp)
        if mf1 > best_f1:
            best_f1 = mf1
            best_tau = float(tau)
    return best_tau, float(best_f1)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="用 gold(dev) 调参并写回 relation_thresholds.json")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--gold-split", type=str, default="dev", choices=["dev"])
    ap.add_argument("--write", action="store_true", help="不加此开关则只打印 best_tau，不写文件")
    args = ap.parse_args()

    root = args.project_root.resolve()
    gold_path = root / "data" / "gold" / f"re_gold_{args.seed_type}_{args.gold_split}.jsonl"
    if not gold_path.is_file():
        raise SystemExit(f"未找到 gold：{gold_path}")
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

    tau, mf1 = _pick_best_tau(y_true, pred_label, pred_prob)
    print(f"seed_type={args.seed_type} best_tau={tau:.4f} best_macro_f1={mf1:.4f} n_eval={len(y_true)} missing_preds={missing}")

    if not args.write:
        return

    thr_path = root / "sources" / "relation_thresholds.json"
    thr = _load_json(thr_path) if thr_path.is_file() else {"version": "v2", "default_min_non_na_prob": 0.5, "by_seed_type": {}}
    by = dict(thr.get("by_seed_type") or {})
    by[args.seed_type] = float(tau)
    thr["version"] = str(thr.get("version") or "v2")
    thr["by_seed_type"] = by
    if "default_min_non_na_prob" not in thr:
        thr["default_min_non_na_prob"] = 0.5
    _write_json(thr_path, thr)
    print(f"已写回阈值：{thr_path}")


if __name__ == "__main__":
    main()

