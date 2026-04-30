from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _clean_compact(s: str) -> str:
    """用于 CRF 字级 token 的“字符序列”视角：去掉空白，保证 token 与匹配串对齐。"""
    return re.sub(r"\s+", "", s or "").strip()


def _find_all_occurrences(haystack: str, needle: str) -> list[int]:
    """返回 needle 在 haystack 中所有（可能重叠的）起始位置。"""
    if not haystack or not needle:
        return []
    out: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        out.append(idx)
        start = idx + 1
    return out


def _bio_tag_span(labels: list[str], start: int, end: int, ent_type: str) -> bool:
    """
    尝试把 [start,end) 位置标为 BIO(ent_type)。
    返回是否成功（若发生跨类型冲突则拒绝该 occurrence）。
    """
    if start < 0 or end <= start or end > len(labels):
        return False

    # 冲突判断：遇到不同 entity type 的非 O 标签就拒绝
    ent_prefix = ("B-" + ent_type, "I-" + ent_type)
    for i in range(start, end):
        cur = labels[i]
        if cur == "O":
            continue
        if cur in ent_prefix or cur == "I-" + ent_type or cur == "B-" + ent_type:
            continue
        # cur 不是同类型：拒绝本次 occurrence
        return False

    # 写入 BIO
    for i in range(start, end):
        labels[i] = (("B-" if i == start else "I-") + ent_type)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="把 data/curated/mentions.jsonl 转为 CRF CoNLL BIO 标注。")
    ap.add_argument("--mentions", default="data/curated/mentions.jsonl")
    ap.add_argument("--out-dir", default="data/ner_crf")
    ap.add_argument("--max-sents", type=int, default=200)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--min-token-len", type=int, default=8)
    ap.add_argument("--seed-type", default="", help="可选：仅处理某个 seed_id（如 turing_person）。不填则全处理")
    args = ap.parse_args()

    mentions_path = Path(args.mentions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 句子级聚合：同一 sentence（同 source_id + sentence_idx + sentence）下合并多个 mention
    # 注意：这里基于 sentence 文本做 key，避免缺少 char offsets 时的复杂对齐。
    bucket: dict[tuple[str, int, str], list[tuple[str, str]]] = defaultdict(list)
    # key: (source_id, sentence_idx, sentence) -> list[(mention, ner_label)]

    max_sents = max(0, int(args.max_sents))
    want_seed = str(args.seed_type or "").strip()

    with mentions_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            if want_seed and str(row.get("seed_id") or "") != want_seed:
                continue
            sent = str(row.get("sentence") or "")
            if not sent.strip():
                continue
            key = (str(row.get("source_id") or ""), int(row.get("sentence_idx") or 0), sent)
            bucket[key].append((str(row.get("mention") or ""), str(row.get("ner_label") or "")))

    # 按 sentence 分批生成 CoNLL
    # 同一 sentence 可能来自多个 seed/行，这里已经合并 mention 了
    sentence_items = list(bucket.items())
    # 稳定顺序：按 source_id + sentence_idx + sentence 前缀
    sentence_items.sort(key=lambda x: (x[0][0], x[0][1], x[0][2][:30]))
    sentence_items = sentence_items[:max_sents] if max_sents else sentence_items

    train_cut = int(len(sentence_items) * float(args.train_ratio))
    train_items = sentence_items[:train_cut]
    dev_items = sentence_items[train_cut:]

    def to_conll_for_items(items: list[Any]) -> str:
        parts: list[str] = []
        for (_source_id, _sentence_idx, sent), mentions in items:
            sent_compact = _clean_compact(sent)
            tokens = list(sent_compact)
            if len(tokens) < int(args.min_token_len):
                continue

            labels = ["O"] * len(tokens)

            # mention token 数越长越优先，减少嵌套/重叠冲突
            proc = []
            for men, lab in mentions:
                lab = str(lab or "").strip().upper()
                men_c = _clean_compact(men)
                if not men_c or not lab or lab == "O":
                    continue
                proc.append((len(men_c), men_c, lab))
            proc.sort(key=lambda x: (-x[0], x[1]))

            for _mlen, men_c, lab in proc:
                occs = _find_all_occurrences(sent_compact, men_c)
                if not occs:
                    continue
                # occurrence 也按起点从左到右写
                occs.sort()
                for start in occs:
                    if len(men_c) == 0:
                        continue
                    end = start + len(men_c)
                    _bio_tag_span(labels, start, end, lab)

            for tok, lab in zip(tokens, labels):
                # CRF CoNLL 读取时会 line.strip() 并对 tok.strip()，因此我们这里保证 tok 不会是纯空白
                parts.append(f"{tok}\t{lab}")
            parts.append("")  # sentence boundary
        return "\n".join(parts).strip() + "\n"

    train_text = to_conll_for_items(train_items)
    dev_text = to_conll_for_items(dev_items)

    (out_dir / "train.conll").write_text(train_text, encoding="utf-8")
    (out_dir / "dev.conll").write_text(dev_text, encoding="utf-8")

    # 可选：写个元信息便于你在文档里解释“标注来源”
    meta = {
        "input": str(mentions_path),
        "note": "弱标注：从 data/curated/mentions.jsonl 的 spaCy 识别 mention + ner_label 自动转换为 CRF BIO。",
        "max_sents": args.max_sents,
        "train_ratio": args.train_ratio,
        "seed_type_filter": want_seed,
    }
    (out_dir / "meta_from_mentions.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {(out_dir / 'train.conll')}  {(out_dir / 'dev.conll')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

