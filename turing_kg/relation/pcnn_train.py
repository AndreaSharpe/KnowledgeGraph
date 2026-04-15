"""PCNN + MIL-Attention 训练：读 bags.jsonl + ds_labels.jsonl，按 seed_type 过滤。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from ..attribution.seed_config import load_seed_entities
from .config_loaders import labels_space_for_seed, load_relation_allowlist
from .ds_labels import read_jsonl
from .pcnn_mil import PCNNMILAttention, bce_loss_bag, multihot_from_labels


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


def prepare_bag_tensors(
    project_root: Path,
    *,
    seed_type: str,
    max_len: int = 256,
    min_freq: int = 1,
) -> tuple[
    list[tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]],
    list[str],
    dict[str, int],
    dict[str, Any],
]:
    """
    返回：
    - bags_data: 每个元素为 (list of (char_ids_long_tensor), pos_e1, pos_e2, y_multihot)
      实际训练时每个 bag 一个 tensor (n, L), pos (n,), y (C,)
    - labels_space
    - char2id
    - meta dict (anchors_by_seed, etc.)
    """
    bags_path = project_root / "data" / "curated" / "bags.jsonl"
    ds_path = project_root / "data" / "curated" / "ds_labels.jsonl"
    allowlist = load_relation_allowlist(project_root)
    seeds = load_seed_entities(project_root)
    anchors_by_seed: dict[str, tuple[str, ...]] = {
        s.seed_id: tuple(s.anchors_zh) + tuple(s.anchors_en) for s in seeds
    }

    ds_map = _load_ds_map(ds_path)
    all_bags = _load_bags(bags_path)

    filtered: list[dict[str, Any]] = [b for b in all_bags if str(b.get("seed_type")) == seed_type]
    sentences_for_vocab: list[str] = []
    for b in filtered:
        for ins in b.get("instances") or []:
            sentences_for_vocab.append(str(ins.get("sentence") or ""))

    char2id = _build_vocab(sentences_for_vocab, min_freq=min_freq)
    canonical_space = labels_space_for_seed(allowlist, seed_type=seed_type, seed_id="")

    if not canonical_space:
        return [], [], char2id, {"anchors_by_seed": anchors_by_seed}

    if not filtered:
        return [], canonical_space, char2id, {"anchors_by_seed": anchors_by_seed}

    bags_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for bag in filtered:
        bid = str(bag.get("bag_id") or "")
        ds = ds_map.get(bid)
        if not ds:
            continue
        seed_id = str(bag.get("seed_id") or "")
        labels_pos = list(ds.get("labels_pos") or [])
        anchors = anchors_by_seed.get(seed_id, ())

        batch = bag_to_model_batch(bag, char2id, max_len, anchors)
        if batch is None:
            continue
        x_batch, p1, p2 = batch

        device_cpu = torch.device("cpu")
        y = multihot_from_labels(labels_pos, canonical_space, device_cpu)

        bags_tensors.append((x_batch, p1, p2, y))

    return bags_tensors, canonical_space, char2id, {"anchors_by_seed": anchors_by_seed}


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

    bags_data, space, char2id, _meta = prepare_bag_tensors(project_root, seed_type=seed_type, max_len=max_len)
    if not bags_data or not space:
        raise RuntimeError(f"无训练样本或关系空间为空：seed_type={seed_type}，请先运行 build_relation_bags / build_ds_labels。")

    num_classes = len(space)
    vocab_size = max(char2id.values()) + 1
    model = PCNNMILAttention(vocab_size, num_classes, pad_id=PAD_ID, att_dim=att_dim).to(dev)

    ys = torch.stack([b[3] for b in bags_data]).to(dev)
    pos_c = ys.sum(dim=0)
    n_bags = float(len(bags_data))
    pos_weight = (n_bags - pos_c) / (pos_c + 1e-6)
    pos_weight = torch.clamp(pos_weight, 0.5, 20.0)

    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        random.shuffle(bags_data)
        total_loss = 0.0
        n_b = 0
        for xb, p1, p2, y in bags_data:
            xb = xb.to(dev)
            p1 = p1.to(dev)
            p2 = p2.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            logits_b, _attn = model.forward_bag(xb, p1, p2)
            loss = bce_loss_bag(logits_b, y, pos_weight=pos_weight)
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
            "aggregation": "mil_attention",
            "att_dim": att_dim,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "max_len": max_len,
            "char2id": char2id,
            "labels_space": space,
            "seed_type": seed_type,
        },
        ckpt_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "seed_type": seed_type,
                "labels_space": space,
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
