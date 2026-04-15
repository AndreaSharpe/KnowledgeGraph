"""多归因 triple 去重/合并：同键合并 reasons，relevance_score 与 link_score 取 max。"""

from __future__ import annotations

import json
from typing import Any


def _parse_reasons(s: str) -> dict[str, Any]:
    if not (s or "").strip():
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _merge_reason_dicts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in parts:
        for k, v in d.items():
            if k not in out:
                out[k] = v
                continue
            old = out[k]
            if isinstance(old, list) and isinstance(v, list):
                out[k] = list(dict.fromkeys([*old, *v]))[:32]
            elif k == "window_expanded":
                out[k] = bool(old) or bool(v)
            elif isinstance(old, (int, float)) and isinstance(v, (int, float)):
                out[k] = max(float(old), float(v))
            else:
                # 保留信息量更大的（字符串长度）
                if isinstance(old, str) and isinstance(v, str) and len(v) > len(old):
                    out[k] = v
    return out


def merge_triple_rows(rows: list[dict[str, str]], *, max_reasons_len: int = 1200) -> list[dict[str, str]]:
    """
    按 (seed_id, sentence_idx, predicate, object_qid 或 literal) 合并行。

    - relevance_score：取数值 max
    - link_score：取数值 max
    - reasons：合并 JSON 对象（列表并集、window_expanded 为或）
    - extraction_method：去重后用 | 连接
    - snippet / evidence_sentence：取较长者（保留更多上下文）
    """
    buckets: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}

    for row in rows:
        seed_id = (row.get("seed_id") or "").strip()
        sent_idx = (row.get("sentence_idx") or "").strip()
        pred = (row.get("predicate") or "").strip()
        oqid = (row.get("object_qid") or "").strip()
        omen = (row.get("object_mention") or "").strip()
        ner = (row.get("ner_label") or "").strip()
        if oqid:
            dedup_key = (seed_id, sent_idx, pred, f"qid:{oqid}")
        else:
            dedup_key = (seed_id, sent_idx, pred, f"lit:{ner}:{omen}")

        buckets.setdefault(dedup_key, []).append(row)

    out: list[dict[str, str]] = []
    for _k, group in sorted(buckets.items(), key=lambda x: x[0]):
        if len(group) == 1:
            out.append(group[0])
            continue

        base = dict(group[0])
        rel_scores: list[float] = []
        link_scores: list[float] = []
        reason_parts: list[dict[str, Any]] = []
        methods: set[str] = set()
        snippets: list[str] = []
        evs: list[str] = []

        for r in group:
            rs = (r.get("relevance_score") or "").strip()
            if rs:
                try:
                    rel_scores.append(float(rs))
                except ValueError:
                    pass
            ls = (r.get("link_score") or "").strip()
            if ls:
                try:
                    link_scores.append(float(ls))
                except ValueError:
                    pass
            reason_parts.append(_parse_reasons(r.get("reasons") or ""))
            m = (r.get("extraction_method") or "").strip()
            if m:
                methods.add(m)
            snippets.append((r.get("snippet") or "").strip())
            evs.append((r.get("evidence_sentence") or "").strip())

        merged_reasons = _merge_reason_dicts(reason_parts)
        reasons_str = json.dumps(merged_reasons, ensure_ascii=False)[:max_reasons_len]

        base["relevance_score"] = f"{max(rel_scores):.4f}" if rel_scores else ""
        base["link_score"] = f"{max(link_scores):.4f}" if link_scores else (base.get("link_score") or "")
        base["reasons"] = reasons_str
        base["extraction_method"] = "|".join(sorted(methods)) if methods else (base.get("extraction_method") or "")
        base["snippet"] = max(snippets, key=len) if snippets else base.get("snippet", "")
        base["evidence_sentence"] = max(evs, key=len) if evs else base.get("evidence_sentence", "")
        out.append(base)

    return out
