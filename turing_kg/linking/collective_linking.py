from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..structured.wikidata_api import iter_claim_item_edges, wbgetentities


@dataclass(frozen=True)
class Candidate:
    qid: str
    score: float
    meta: dict[str, Any]


@dataclass(frozen=True)
class MentionNode:
    key: str
    sentence_idx: int
    mention: str
    candidates: tuple[Candidate, ...]
    chosen_qid: str | None


@dataclass(frozen=True)
class CollectiveConfig:
    enabled: bool = True
    window_sentences: int = 2
    top_k_candidates: int = 6
    lambda_coherence: float = 0.35
    coherence_props: tuple[str, ...] = ("P31", "P279", "P361", "P17", "P276", "P1416", "P166")
    max_entities_to_fetch: int = 120


def _jaccard_set(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _neighbors_from_claims(claims: dict[str, Any], *, only_props: set[str]) -> set[str]:
    out: set[str] = set()
    for p, q in iter_claim_item_edges(claims or {}):
        if p in only_props:
            out.add(q)
    return out


def fetch_neighbor_sets(
    qids: list[str],
    *,
    props: tuple[str, ...],
    languages: str = "zh|en",
    max_entities: int = 120,
) -> dict[str, set[str]]:
    """
    为 coherence 提取一跳邻域（按 props 过滤），并利用 `wbgetentities` 自带缓存/退避。
    """
    uniq = [q for q in dict.fromkeys([q for q in qids if q and q.startswith("Q")])]
    if not uniq:
        return {}
    uniq = uniq[: max(1, int(max_entities))]
    ents = wbgetentities(uniq, props="claims", languages=languages)
    only = set(props)
    out: dict[str, set[str]] = {}
    for q in uniq:
        claims = (ents.get(q) or {}).get("claims") or {}
        out[q] = _neighbors_from_claims(claims, only_props=only)
    return out


def collective_decode_window(
    nodes: list[MentionNode],
    *,
    neighbor_sets: dict[str, set[str]],
    lam: float,
    max_iters: int = 3,
) -> dict[str, dict[str, Any]]:
    """
    对一个窗口内的 mentions 做贪心坐标上升解码。

    返回：node.key -> {chosen_qid, local_score, global_score, total_score, changed_from}
    """
    # 初始化：local 最优
    sel: dict[str, Candidate] = {}
    for n in nodes:
        best = max(n.candidates, key=lambda c: c.score, default=None)
        if best is not None:
            sel[n.key] = best

    def coh(q1: str, q2: str) -> float:
        if not q1 or not q2 or q1 == q2:
            return 0.0
        return _jaccard_set(neighbor_sets.get(q1, set()), neighbor_sets.get(q2, set()))

    keys = [n.key for n in nodes if n.key in sel]
    for _it in range(max_iters):
        improved = False
        for n in nodes:
            if n.key not in sel:
                continue
            cur = sel[n.key]
            best = cur
            best_obj = cur.score
            # coherence with others
            for k in keys:
                if k == n.key:
                    continue
                best_obj += lam * coh(cur.qid, sel[k].qid)

            for cand in n.candidates:
                obj = cand.score
                for k in keys:
                    if k == n.key:
                        continue
                    obj += lam * coh(cand.qid, sel[k].qid)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best = cand
            if best.qid != cur.qid:
                sel[n.key] = best
                improved = True
        if not improved:
            break

    # 汇总 global 分数贡献（相对 local 的 coherence 部分）
    out: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if n.key not in sel:
            continue
        chosen = sel[n.key]
        gsum = 0.0
        for k in keys:
            if k == n.key:
                continue
            gsum += coh(chosen.qid, sel[k].qid)
        out[n.key] = {
            "chosen_qid": chosen.qid,
            "local_score": float(chosen.score),
            "global_score": float(lam * gsum),
            "total_score": float(chosen.score + lam * gsum),
            "changed_from": n.chosen_qid if n.chosen_qid and n.chosen_qid != chosen.qid else None,
        }
    return out

