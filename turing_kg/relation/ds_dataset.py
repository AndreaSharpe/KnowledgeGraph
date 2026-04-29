"""
将 bags.jsonl + ds_labels.jsonl 转成更贴近经典 DS 的 bag-level 多分类数据集（含 NA/no_relation）。

输出格式（JSONL，每行一个 bag）：
- bag_id, seed_type, seed_id, subject_qid, object_qid, instances, label
- split: train/dev/test

设定：
- label 空间：["NA"] + allowlist[seed_type]
- 若 labels_pos 为空 => NA
- 若 labels_pos 多于 1：默认丢弃（避免多标签污染多分类基线）；可配置为取优先级或取第一个。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config_loaders import labels_space_for_seed, load_relation_allowlist
from .ds_labels import read_jsonl


def _split_assign(bag_ids: list[str], *, seed: int, p_train: float, p_dev: float) -> dict[str, str]:
    """按 bag_id 做可复现的 train/dev/test 切分映射。"""
    rnd = random.Random(seed)
    ids = list(bag_ids)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * p_train)
    n_dev = int(n * p_dev)
    out: dict[str, str] = {}
    for i, bid in enumerate(ids):
        if i < n_train:
            out[bid] = "train"
        elif i < n_train + n_dev:
            out[bid] = "dev"
        else:
            out[bid] = "test"
    return out


def build_multiclass_dataset_rows(
    project_root: Path,
    *,
    seed_type: str,
    na_ratio: float = 1.0,
    max_na_when_no_pos: int = 200,
    split_seed: int = 42,
    p_train: float = 0.8,
    p_dev: float = 0.1,
    drop_multi_pos: bool = True,
    keep_fields: Iterable[str] = ("bag_id", "seed_type", "seed_id", "subject_qid", "object_qid", "instances"),
) -> list[dict[str, Any]]:
    """将 bags+ds_labels 转成 DS 多分类 bag 数据（含 NA），返回行列表。"""
    curated = project_root / "data" / "curated"
    bags = read_jsonl(curated / "bags.jsonl")
    ds = read_jsonl(curated / "ds_labels.jsonl")
    ds_by_bag = {str(r.get("bag_id") or ""): r for r in ds if str(r.get("bag_id") or "")}

    allowlist = load_relation_allowlist(project_root)
    space = labels_space_for_seed(allowlist, seed_type=seed_type, seed_id="")
    if not space:
        return []

    # 先构造正/负候选
    pos_rows: list[dict[str, Any]] = []
    na_rows: list[dict[str, Any]] = []

    for b in bags:
        if str(b.get("seed_type") or "") != seed_type:
            continue
        bid = str(b.get("bag_id") or "")
        d = ds_by_bag.get(bid)
        if not bid or not d:
            continue

        labels_pos = [str(x) for x in (d.get("labels_pos") or []) if str(x)]
        labels_pos = [p for p in labels_pos if p in set(space)]

        if len(labels_pos) > 1 and drop_multi_pos:
            continue

        base = {k: b.get(k) for k in keep_fields}
        if labels_pos:
            row = dict(base)
            row["label"] = labels_pos[0]
            pos_rows.append(row)
        else:
            row = dict(base)
            row["label"] = "NA"
            na_rows.append(row)

    # NA 下采样，控制 NA:POS 比例
    if na_ratio <= 0:
        na_keep = []
    else:
        target = int(len(pos_rows) * na_ratio)
        rnd = random.Random(split_seed)
        na_keep = list(na_rows)
        rnd.shuffle(na_keep)
        if target > 0:
            na_keep = na_keep[:target]
        else:
            # 经典 DS 里 POS=0 时无法训练，但为了诊断/标注仍保留一小批 NA 行
            na_keep = na_keep[: max(0, int(max_na_when_no_pos))]

    rows = pos_rows + na_keep
    # 分割按 bag_id（可复现）
    split_map = _split_assign([str(r["bag_id"]) for r in rows], seed=split_seed, p_train=p_train, p_dev=p_dev)
    for r in rows:
        r["split"] = split_map.get(str(r["bag_id"]), "train")
        r["label_space"] = ["NA"] + list(space)
    rows.sort(key=lambda x: (x.get("split", ""), x.get("bag_id", "")))
    return rows


def write_multiclass_dataset_jsonl(project_root: Path, *, seed_type: str, rows: list[dict[str, Any]]) -> Path:
    """写入 data/curated/re_ds_dataset_{seed_type}.jsonl（每行一个 bag）。"""
    out = project_root / "data" / "curated" / f"re_ds_dataset_{seed_type}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def build_and_write_multiclass_dataset(
    project_root: Path,
    *,
    seed_type: str,
    na_ratio: float = 1.0,
    max_na_when_no_pos: int = 200,
    split_seed: int = 42,
    p_train: float = 0.8,
    p_dev: float = 0.1,
    drop_multi_pos: bool = True,
) -> tuple[Path, int]:
    """一键构建并落盘 DS 多分类数据集，返回 (路径, 行数)。"""
    rows = build_multiclass_dataset_rows(
        project_root,
        seed_type=seed_type,
        na_ratio=na_ratio,
        max_na_when_no_pos=max_na_when_no_pos,
        split_seed=split_seed,
        p_train=p_train,
        p_dev=p_dev,
        drop_multi_pos=drop_multi_pos,
    )
    p = write_multiclass_dataset_jsonl(project_root, seed_type=seed_type, rows=rows)
    return p, len(rows)

