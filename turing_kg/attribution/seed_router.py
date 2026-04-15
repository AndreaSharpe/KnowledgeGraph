from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .seed_config import AttributionConfig, SeedSpec


@dataclass(frozen=True)
class SeedAttribution:
    seed_id: str
    seed_qid: str
    score: float
    reasons: dict[str, Any]


@dataclass(frozen=True)
class SentenceAttribution:
    sentence: str
    assigned: tuple[SeedAttribution, ...]


def _hits(text: str, patterns: tuple[str, ...]) -> list[str]:
    t = text.strip()
    out: list[str] = []
    for p in patterns:
        if not p:
            continue
        if p in t:
            out.append(p)
    return out


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def route_sentences(
    sentences: list[str],
    *,
    seeds: list[SeedSpec],
    cfg: AttributionConfig,
) -> list[SentenceAttribution]:
    """
    规则+线性加权句子路由（可审计）。

    说明：
    - 仅实现最小可行机制：anchor 命中、trigger 命中、邻近 anchor 弱加分
    - 语言不在此处判定：seed 同时可带 zh/en anchors；调用方可按文本语言裁剪 seeds
    """
    w = cfg.score_weights or {}
    w_anchor = float(w.get("anchor_hit", 1.0))
    w_trigger = float(w.get("trigger_hit", 0.35))
    w_near = float(w.get("nearby_anchor_bonus", 0.15))

    sents = [_normalize_ws(s) for s in sentences if _normalize_ws(s)]
    # 先找每句命中的 anchors（用于距离弱证据）
    anchor_hits_by_seed: list[list[list[str]]] = []  # seed -> sent -> hits
    for sd in seeds:
        hits = []
        pats = tuple(sd.anchors_zh) + tuple(sd.anchors_en)
        for s in sents:
            hits.append(_hits(s, pats))
        anchor_hits_by_seed.append(hits)

    out: list[SentenceAttribution] = []
    for i, s in enumerate(sents):
        scored: list[SeedAttribution] = []
        for si, sd in enumerate(seeds):
            anchor_hits = anchor_hits_by_seed[si][i]
            trigger_hits = _hits(s, tuple(sd.triggers_zh) + tuple(sd.triggers_en))

            score = 0.0
            score += w_anchor * float(len(anchor_hits))
            score += w_trigger * float(len(trigger_hits))

            # 邻近 anchor 弱证据：若相邻句出现该 seed 的 anchor
            near = 0
            if w_near > 0:
                for j in (i - 1, i + 1):
                    if 0 <= j < len(sents) and anchor_hits_by_seed[si][j]:
                        near += 1
                if near:
                    score += w_near

            if score <= 0:
                continue

            reasons: dict[str, Any] = {}
            if anchor_hits:
                reasons["anchor_hits"] = anchor_hits[: cfg.max_reasons_items]
            if trigger_hits:
                reasons["trigger_hits"] = trigger_hits[: cfg.max_reasons_items]
            if near:
                reasons["nearby_anchor"] = True

            scored.append(
                SeedAttribution(
                    seed_id=sd.seed_id,
                    seed_qid=sd.qid,
                    score=score,
                    reasons=reasons,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        assigned = tuple([x for x in scored if x.score >= cfg.min_score][: max(cfg.top_k, 1)])
        out.append(SentenceAttribution(sentence=s, assigned=assigned))
    return out

