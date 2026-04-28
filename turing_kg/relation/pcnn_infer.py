"""加载 PCNN checkpoint，对 bags 推理，写入 data/curated/re_predictions.jsonl。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..attribution.seed_config import load_seed_entities
from .config_loaders import load_relation_thresholds
from .ds_labels import read_jsonl
from .pcnn_mil import PCNNSelectiveAttention
from .pcnn_train import PAD_ID, bag_to_model_batch


def load_checkpoint(project_root: Path, seed_type: str, ckpt_path: Path | None = None) -> dict[str, Any]:
    ckpt_path = ckpt_path or (project_root / "models" / "relation_pcnn" / f"pcnn_{seed_type}.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint：{ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def build_model_from_ckpt(ckpt: dict[str, Any], device: torch.device) -> PCNNSelectiveAttention:
    char2id: dict[str, int] = ckpt["char2id"]
    vocab_size = int(ckpt["vocab_size"])
    num_classes = int(ckpt["num_classes"])
    att_dim = int(ckpt.get("att_dim", 128))
    if ckpt.get("setting") != "ds_multiclass_selective_attention":
        raise RuntimeError("checkpoint 不是经典 DS 多分类 selective attention，请重新训练。")
    model = PCNNSelectiveAttention(vocab_size, num_classes, pad_id=PAD_ID, att_dim=att_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer_bags(
    project_root: Path,
    *,
    seed_type: str,
    ckpt_path: Path | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(project_root, seed_type, ckpt_path)
    char2id: dict[str, int] = ckpt["char2id"]
    max_len = int(ckpt["max_len"])

    seeds = load_seed_entities(project_root)
    anchors_by_seed: dict[str, tuple[str, ...]] = {
        s.seed_id: tuple(s.anchors_zh) + tuple(s.anchors_en) for s in seeds
    }

    bags_path = project_root / "data" / "curated" / "bags.jsonl"
    all_bags = read_jsonl(bags_path)
    filtered = [b for b in all_bags if str(b.get("seed_type")) == seed_type]

    model = build_model_from_ckpt(ckpt, dev)
    label_space_raw = ckpt.get("label_space")
    if not isinstance(label_space_raw, list) or not label_space_raw:
        raise RuntimeError(
            "checkpoint 缺少 label_space，说明它不是当前经典 DS 多分类模型的产物。"
            "请先运行 scripts/build_re_ds_dataset.py 与 scripts/train_relation_pcnn.py 重新训练。"
        )
    label_space: list[str] = [str(x) for x in label_space_raw]
    thr_cfg = load_relation_thresholds(project_root)

    out_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for bag in filtered:
            bid = str(bag.get("bag_id") or "")
            seed_id = str(bag.get("seed_id") or "")
            subject_qid = str(bag.get("subject_qid") or "")
            object_qid = str(bag.get("object_qid") or "")
            anchors = anchors_by_seed.get(seed_id, ())

            batch = bag_to_model_batch(bag, char2id, max_len, anchors)
            if batch is None:
                continue
            xb, p1, p2 = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
            logits_b, attn = model.forward_bag(xb, p1, p2)
            probs = torch.softmax(logits_b, dim=0).cpu()
            attn_cpu = attn.detach().cpu()  # (C, n_inst)

            pred_idx = int(probs.argmax().item())
            pred_label = str(label_space[pred_idx]) if 0 <= pred_idx < len(label_space) else "NA"
            pred_prob = float(probs[pred_idx].item()) if 0 <= pred_idx < probs.numel() else 0.0

            instances_meta = list(bag.get("instances") or [])
            best_i = int(attn_cpu[pred_idx].argmax().item()) if attn_cpu.numel() else 0
            ev_sentence = ""
            ev_idx = -1
            if 0 <= best_i < len(instances_meta):
                ev_sentence = str(instances_meta[best_i].get("sentence") or "")
                ev_idx = int(instances_meta[best_i].get("sentence_idx", -1))
            attn_at_top = (
                float(attn_cpu[pred_idx, best_i].item()) if attn_cpu.numel() and pred_idx < attn_cpu.shape[0] else 0.0
            )

            # 兼容导出：仍输出“候选列表”，但 label 可能是 NA 或 Pxxx
            preds: list[dict[str, Any]] = []
            for ci, lb in enumerate(label_space):
                score = float(probs[ci].item())
                # 多分类下阈值策略后续统一在 mil_ingest 里处理；这里先把阈值原样携带
                preds.append({"label": str(lb), "score": score})

            out_rows.append(
                {
                    "bag_id": bid,
                    "seed_id": seed_id,
                    "seed_type": seed_type,
                    "subject_qid": subject_qid,
                    "object_qid": object_qid,
                    "source_id": str(bag.get("source_id") or ""),
                    "source_url": str(bag.get("source_url") or ""),
                    "citation_key": str(bag.get("citation_key") or ""),
                    "model": f"pcnn_ds_mc_selectatt_{seed_type}",
                    "predicted_label": pred_label,
                    "predicted_prob": pred_prob,
                    "evidence": {
                        "top_sentence_idx": ev_idx,
                        "top_sentence": ev_sentence[:500],
                        "top_sentence_score": attn_at_top,
                        "attention_instance_idx": best_i,
                        "predicted_label": pred_label,
                    },
                    "class_probs": preds,
                    "thresholds": thr_cfg,
                }
            )

    out_rows.sort(key=lambda r: r.get("bag_id", ""))
    return out_rows


def write_re_predictions_merged_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> Path:
    """合并写入 re_predictions.jsonl（infer_all 使用）。"""
    out = project_root / "data" / "curated" / "re_predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def write_re_predictions_pcnn_jsonl(project_root: Path, rows: list[dict[str, Any]], seed_type: str) -> Path:
    """单 seed_type 推理结果，避免覆盖其它类型。"""
    out = project_root / "data" / "curated" / f"re_predictions_pcnn_{seed_type}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def infer_all_available(
    project_root: Path,
    *,
    device: str | None = None,
) -> tuple[Path, int]:
    """对每个存在 checkpoint 的 seed_type 推理并合并写入 re_predictions.jsonl。"""
    merged: list[dict[str, Any]] = []
    base = project_root / "models" / "relation_pcnn"
    for st in ("Person", "Concept", "Award"):
        ckpt = base / f"pcnn_{st}.pt"
        if not ckpt.is_file():
            continue
        merged.extend(infer_bags(project_root, seed_type=st, ckpt_path=ckpt, device=device))
    merged.sort(key=lambda r: (r.get("seed_type", ""), r.get("bag_id", "")))
    path = write_re_predictions_merged_jsonl(project_root, merged)
    return path, len(merged)


def infer_and_write(
    project_root: Path,
    *,
    seed_type: str,
    ckpt_path: Path | None = None,
    device: str | None = None,
) -> tuple[Path, int]:
    rows = infer_bags(
        project_root,
        seed_type=seed_type,
        ckpt_path=ckpt_path,
        device=device,
    )
    path = write_re_predictions_pcnn_jsonl(project_root, rows, seed_type)
    return path, len(rows)
