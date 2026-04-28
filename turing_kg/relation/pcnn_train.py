"""经典 DS 训练：PCNN + selective attention（Lin et al.）+ bag-level 多分类（含 NA）。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..attribution.seed_config import load_seed_entities
from .ds_labels import read_jsonl
from .pcnn_mil import PCNNSelectiveAttention


PAD_CHAR = "<pad>"
UNK_CHAR = "<unk>"
PAD_ID = 0
UNK_ID = 1


def _load_bags(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_ds_map(path: Path) -> dict[str, dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        bid = str(row.get("bag_id") or "")
        if bid:
            m[bid] = row
    return m


def _anchor_start(sentence: str, anchors: tuple[str, ...]) -> int:
    best = len(sentence)
    ok = False
    for a in anchors:
        a = (a or "").strip()
        if len(a) < 1:
            continue
        i = sentence.find(a)
        if i >= 0 and i < best:
            best = i
            ok = True
    return best if ok else 0


def _mention_start(sentence: str, mention: str) -> int:
    m = (mention or "").strip()
    if not m:
        return min(1, max(0, len(sentence) - 1))
    i = sentence.find(m)
    return i if i >= 0 else min(1, max(0, len(sentence) - 1))


def _build_vocab(sentences: list[str], min_freq: int = 1) -> dict[str, int]:
    freq: dict[str, int] = {}
    for s in sentences:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    char2id: dict[str, int] = {PAD_CHAR: PAD_ID, UNK_CHAR: UNK_ID}
    nxt = UNK_ID + 1
    for ch, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c < min_freq:
            continue
        if ch not in char2id:
            char2id[ch] = nxt
            nxt += 1
    return char2id


def bag_to_model_batch(
    bag: dict[str, Any],
    char2id: dict[str, int],
    max_len: int,
    anchors: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """单 bag -> (n, L), (n,), (n,)，无实例则 None。"""
    inst_list: list[tuple[list[int], int, int]] = []
    for ins in bag.get("instances") or []:
        sent = str(ins.get("sentence") or "")
        if not sent.strip():
            continue
        ments = ins.get("mentions") or {}
        mention = str(ments.get("object_mention") or "")
        e1 = _anchor_start(sent, anchors)
        e2 = _mention_start(sent, mention)
        if e1 > e2:
            e1, e2 = e2, e1
        if e1 == e2 and len(sent) > 1:
            e2 = min(e1 + 1, len(sent) - 1)
        ids = _encode(sent, char2id, max_len)
        le = min(len(sent), max_len)
        e1 = min(e1, le - 1)
        e2 = min(e2, le - 1)
        inst_list.append((ids, e1, e2))
    if not inst_list:
        return None
    char_rows = [torch.tensor(x[0], dtype=torch.long) for x in inst_list]
    p1 = torch.tensor([x[1] for x in inst_list], dtype=torch.long)
    p2 = torch.tensor([x[2] for x in inst_list], dtype=torch.long)
    return torch.stack(char_rows, dim=0), p1, p2


def _encode(s: str, char2id: dict[str, int], max_len: int) -> list[int]:
    unk = char2id.get(UNK_CHAR, UNK_ID)
    pad = char2id.get(PAD_CHAR, PAD_ID)
    ids = [char2id.get(ch, unk) for ch in s[:max_len]]
    while len(ids) < max_len:
        ids.append(pad)
    return ids[:max_len]


def prepare_dataset_tensors(
    project_root: Path,
    *,
    seed_type: str,
    max_len: int = 256,
    min_freq: int = 1,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    list[str],  # label_space（含 NA）
    dict[str, int],
    dict[str, Any],
]:
    """读 data/curated/re_ds_dataset_{seed_type}.jsonl 并转成训练张量（仅 split=train）。"""
    ds_path = project_root / "data" / "curated" / f"re_ds_dataset_{seed_type}.jsonl"
    seeds = load_seed_entities(project_root)
    anchors_by_seed: dict[str, tuple[str, ...]] = {
        s.seed_id: tuple(s.anchors_zh) + tuple(s.anchors_en) for s in seeds
    }
    rows = read_jsonl(ds_path)
    if not rows:
        return [], [], {PAD_CHAR: PAD_ID, UNK_CHAR: UNK_ID}, {"anchors_by_seed": anchors_by_seed, "dataset_path": str(ds_path)}

    # vocab from all splits
    sents: list[str] = []
    for r in rows:
        for ins in r.get("instances") or []:
            sents.append(str(ins.get("sentence") or ""))
    char2id = _build_vocab(sents, min_freq=min_freq)

    label_space = list(rows[0].get("label_space") or [])
    if not label_space or label_space[0] != "NA":
        raise RuntimeError(f"dataset 缺少 label_space 或未包含 NA：{ds_path}")
    label2idx = {str(lb): i for i, lb in enumerate(label_space)}

    data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for r in rows:
        if str(r.get("split") or "") != "train":
            continue
        seed_id = str(r.get("seed_id") or "")
        anchors = anchors_by_seed.get(seed_id, ())
        batch = bag_to_model_batch(r, char2id, max_len, anchors)
        if batch is None:
            continue
        y = str(r.get("label") or "NA")
        if y not in label2idx:
            continue
        data.append((batch[0], batch[1], batch[2], int(label2idx[y])))

    return data, label_space, char2id, {"anchors_by_seed": anchors_by_seed, "dataset_path": str(ds_path)}


def train_pcnn_mil(
    project_root: Path,
    *,
    seed_type: str,
    epochs: int = 15,
    lr: float = 1e-3,
    max_len: int = 256,
    att_dim: int = 128,
    device: str | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    torch.manual_seed(seed)

    train_data, label_space, char2id, meta = prepare_dataset_tensors(project_root, seed_type=seed_type, max_len=max_len)
    if not train_data or not label_space:
        raise RuntimeError(
            f"无训练样本或 label_space 为空：seed_type={seed_type}。请先运行 scripts/build_re_ds_dataset.py --seed-type {seed_type}。"
        )

    num_classes = len(label_space)
    vocab_size = max(char2id.values()) + 1
    model = PCNNSelectiveAttention(vocab_size, num_classes, pad_id=PAD_ID, att_dim=att_dim).to(dev)

    # class weights
    y_counts = torch.zeros(num_classes, dtype=torch.float)
    for _xb, _p1, _p2, yi in train_data:
        y_counts[int(yi)] += 1.0
    w = (y_counts.sum() - y_counts) / (y_counts + 1e-6)
    w = torch.clamp(w, 0.5, 20.0).to(dev)

    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        n_b = 0
        for xb, p1, p2, yi in train_data:
            xb = xb.to(dev)
            p1 = p1.to(dev)
            p2 = p2.to(dev)
            y = torch.tensor(int(yi), dtype=torch.long, device=dev)
            opt.zero_grad()
            logits_b, _attn = model.forward_bag(xb, p1, p2)
            loss = F.cross_entropy(logits_b.unsqueeze(0), y.unsqueeze(0), weight=w)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_b += 1
        avg = total_loss / max(n_b, 1)
        print(f"epoch {ep+1}/{epochs} loss={avg:.4f}")

    out_dir = out_dir or (project_root / "models" / "relation_pcnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"pcnn_{seed_type}.pt"
    meta_path = out_dir / f"pcnn_{seed_type}_meta.json"

    torch.save(
        {
            "model_state": model.state_dict(),
            "setting": "ds_multiclass_selective_attention",
            "att_dim": att_dim,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "max_len": max_len,
            "char2id": char2id,
            "label_space": label_space,
            "seed_type": seed_type,
            "meta": meta,
        },
        ckpt_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "seed_type": seed_type,
                "label_space": label_space,
                "max_len": max_len,
                "vocab_size": vocab_size,
                "checkpoint": str(ckpt_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"checkpoint": str(ckpt_path), "meta": str(meta_path), "device": str(dev)}


def run_train_from_cli(project_root: Path, **kwargs: Any) -> dict[str, Any]:
    return train_pcnn_mil(project_root, **kwargs)
