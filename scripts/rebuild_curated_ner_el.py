#!/usr/bin/env python3
"""
只重建 NER/EL 中间层（mentions / candidates / resolved）。

输入（必须已存在）：
- data/processed/sentences.jsonl
- data/processed/routing.jsonl

输出（覆盖写）：
- data/curated/mentions.jsonl
- data/curated/candidates.jsonl
- data/curated/resolved.jsonl

说明：
- 不入图、不导出 triples，因此比 run.py --mode full 快。
- 若 entity_map.csv 覆盖充分，可大幅减少联网候选检索。
- 默认会为可链接实体写 candidates（可能联网）；可用 --no-candidates 关闭以进一步加速。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _truncate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="仅重建 NER/EL 中间层（mentions/candidates/resolved）")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--no-candidates", action="store_true", help="不生成 candidates.jsonl（更快，仍生成 resolved）")
    ap.add_argument("--seed-id", type=str, default="", help="只重建指定 seed_id（例如 turing_machine）")
    ap.add_argument("--max-sources", type=int, default=0, help="只处理前 N 个 source（用于快速回归）")
    args = ap.parse_args()
    root: Path = args.project_root.resolve()

    processed = root / "data" / "processed"
    curated = root / "data" / "curated"
    sent_path = processed / "sentences.jsonl"
    route_path = processed / "routing.jsonl"
    if not sent_path.is_file() or not route_path.is_file():
        raise SystemExit("缺少 data/processed/sentences.jsonl 或 routing.jsonl；请先运行 run.py（任意模式生成 processed）。")

    # imports（避免脚本启动时触发重依赖）
    from turing_kg.attribution.seed_config import (
        load_entity_linking_config,
        load_extraction_profile,
        load_seed_entities,
        pick_ner_backend_for_source,
    )
    from turing_kg.config import ROOT_ENTITY_QID
    from turing_kg.extraction.ner_link import LinkedSpan, extract_linked_spans_from_sentences
    from turing_kg.io.sources_io import load_entity_map
    from turing_kg.io.stage_io import write_candidates_jsonl, write_mentions_jsonl, write_resolved_jsonl
    from turing_kg.linking.entity_linking import link_mention_to_qid, link_mention_with_candidates

    # 读 sentences（按 source_id 聚合，保留元信息）
    sent_rows = _read_jsonl(sent_path)
    by_source_sent: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    source_meta: dict[str, dict[str, str]] = {}
    for r in sent_rows:
        sid = str(r.get("source_id") or "")
        si = int(r.get("sentence_idx", -1))
        if not sid or si < 0:
            continue
        by_source_sent[sid][si] = r
        if sid not in source_meta:
            source_meta[sid] = {
                "source_url": str(r.get("source_url") or ""),
                "source_label": str(r.get("source_label") or ""),
                "citation_key": str(r.get("citation_key") or ""),
            }

    # 读 routing（按 source_id → seed_id → sentence_idx）
    route_rows = _read_jsonl(route_path)
    by_source_seed_idxs: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for rr in route_rows:
        sid = str(rr.get("source_id") or "")
        si = int(rr.get("sentence_idx", -1))
        if not sid or si < 0:
            continue
        for a in rr.get("assigned") or []:
            seed_id = str(a.get("seed_id") or "")
            if seed_id:
                by_source_seed_idxs[sid][seed_id].add(si)

    # configs
    emap = load_entity_map(root / "sources" / "entity_map.csv")
    seeds = load_seed_entities(root)
    ex_profile = load_extraction_profile(root)
    # 目前仅用于读取配置（与 build.py 保持一致的入口）；本脚本不做 collective linking。
    _ = load_entity_linking_config(root)
    ner_cfg_path = root / "sources" / "ner_link_config.json"
    ner_cfg = json.loads(ner_cfg_path.read_text(encoding="utf-8")) if ner_cfg_path.is_file() else {}
    min_ls = float(ner_cfg.get("min_link_score", 0.14))

    # 清空 curated 三个文件（仅这三个）
    _truncate(curated / "mentions.jsonl")
    _truncate(curated / "candidates.jsonl")
    _truncate(curated / "resolved.jsonl")

    n_mentions = 0
    n_candidates = 0
    n_resolved = 0
    by_seed_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # 主循环：按 source 处理（避免一次性堆太多内存）
    source_ids = list(by_source_sent.keys())
    if args.max_sources and args.max_sources > 0:
        source_ids = source_ids[: int(args.max_sources)]
    for si_source, source_id in enumerate(source_ids):
        sents_map = by_source_sent[source_id]
        meta = source_meta.get(source_id, {})
        url = meta.get("source_url", "")
        label = meta.get("source_label", "")
        cite_key = meta.get("citation_key", "")
        backend = pick_ner_backend_for_source(ex_profile, url or source_id)

        if si_source % 8 == 0:
            print(f"[{si_source+1}/{len(source_ids)}] processing source={label or source_id} backend={backend}")

        for sd in seeds:
            if args.seed_id and sd.seed_id != args.seed_id:
                continue
            idxs = sorted(by_source_seed_idxs.get(source_id, {}).get(sd.seed_id, set()))
            # 与 build.py 保持一致：若该 source 完全没有路由到任何 seed，
            # 则兜底把全篇句子归到 turing_person，避免 NER/EL 重建产物为空。
            if not idxs and sd.seed_id == "turing_person":
                # 仅在该 source 对所有 seed 都为空时兜底
                any_assigned = any(by_source_seed_idxs.get(source_id, {}).get(s.seed_id) for s in seeds)
                if not any_assigned:
                    idxs = sorted(sents_map.keys())
            if not idxs:
                continue
            seed_items = []
            for i in idxs:
                row = sents_map.get(i)
                if not row:
                    continue
                seed_items.append((i, str(row.get("sentence") or "")))
            if not seed_items:
                continue

            root_qid = (sd.qid or "").strip()
            if not root_qid:
                # 兼容：从 entity_map 猜
                guess = (sd.anchors_zh[0] if sd.anchors_zh else (sd.anchors_en[0] if sd.anchors_en else "")).strip()
                row = emap.get(guess) or emap.get(guess.lower())
                if row and row.get("wikidata_id"):
                    root_qid = row["wikidata_id"]
            if not root_qid:
                continue

            # 抽 spans
            spans: list[LinkedSpan] = []
            try:
                if backend == "spacy":
                    spans = extract_linked_spans_from_sentences(
                        seed_items,
                        entity_map=emap,
                        seed_anchors_zh=list(sd.anchors_zh),
                        seed_anchors_en=list(sd.anchors_en),
                        min_link_score=min_ls,
                        source_label=label,
                    )
                else:
                    mp = (root / ex_profile.crf_model_path).resolve()
                    if not mp.is_file():
                        spans = extract_linked_spans_from_sentences(
                            seed_items,
                            entity_map=emap,
                            seed_anchors_zh=list(sd.anchors_zh),
                            seed_anchors_en=list(sd.anchors_en),
                            min_link_score=min_ls,
                            source_label=label,
                        )
                    else:
                        from turing_kg.extraction.crf_ner import extract_entities_with_crf_zh

                        seed_sents = [s for _i, s in seed_items]
                        ents = extract_entities_with_crf_zh(seed_sents, model_path=mp)
                        seen = set()
                        for e in ents:
                            men = (e.mention or "").strip()
                            if len(men) < 2:
                                continue
                            obj_qid = ""
                            sc = 1.0
                            if e.label in ("PER", "ORG", "LOC"):
                                qid, sc2 = link_mention_to_qid(
                                    men,
                                    e.evidence_sentence,
                                    languages=("zh", "en"),
                                    min_score=min_ls,
                                    entity_map_override=emap,
                                )
                                if qid and qid != ROOT_ENTITY_QID:
                                    obj_qid = qid
                                    sc = sc2
                            k = (obj_qid, men.lower(), e.label)
                            if k in seen:
                                continue
                            seen.add(k)
                            sent_idx = -1
                            for ii, ss in seed_items:
                                if ss.strip() == (e.evidence_sentence or "").strip():
                                    sent_idx = ii
                                    break
                            spans.append(
                                LinkedSpan(
                                    sentence_idx=int(sent_idx),
                                    object_qid=str(obj_qid),
                                    mention=men,
                                    snippet=f"[{label}] {(e.evidence_sentence or '')[:280]}",
                                    score=float(sc),
                                    ner_label=str(e.label),
                                    context="crf",
                                )
                            )
            except ImportError as e:
                warnings.warn(str(e), stacklevel=1)
                spans = []

            # 过滤自环
            spans = [sp for sp in spans if sp.object_qid != root_qid]

            mentions_rows: list[dict[str, Any]] = []
            candidates_rows: list[dict[str, Any]] = []
            resolved_rows: list[dict[str, Any]] = []
            seen_key: set[tuple[str, str, int, str, str]] = set()

            for sp in spans:
                # 句子原文
                sent_row = sents_map.get(int(sp.sentence_idx))
                sent_text = str(sent_row.get("sentence") or "") if sent_row else ""

                mk = (source_id, sd.seed_id, int(sp.sentence_idx), sp.mention, sp.ner_label)
                if mk in seen_key:
                    continue
                seen_key.add(mk)

                mentions_rows.append(
                    {
                        "source_id": source_id,
                        "source_url": url,
                        "source_label": label,
                        "citation_key": cite_key,
                        "seed_id": sd.seed_id,
                        "seed_qid": root_qid,
                        "sentence_idx": int(sp.sentence_idx),
                        "sentence": sent_text,
                        "mention": sp.mention,
                        "ner_label": sp.ner_label,
                        "ner_backend": backend,
                        "mention_context": sp.context,
                        "char_start": None,
                        "char_end": None,
                    }
                )

                # literal/no-link
                if not sp.object_qid:
                    resolved_rows.append(
                        {
                            "source_id": source_id,
                            "source_url": url,
                            "source_label": label,
                            "citation_key": cite_key,
                            "seed_id": sd.seed_id,
                            "seed_qid": root_qid,
                            "sentence_idx": int(sp.sentence_idx),
                            "mention": sp.mention,
                            "ner_label": sp.ner_label,
                            "chosen_qid": "",
                            "scores": {"local": float(sp.score), "global": 0.0, "total": float(sp.score)},
                            "reasons": {
                                "local": {"link_score": float(sp.score), "source": "literal_no_wikidata_link"},
                                "global": {},
                                "prior": None,
                            },
                        }
                    )
                    continue

                # 走 candidates（可关闭）
                chosen_qid = str(sp.object_qid)
                chosen_score = float(sp.score)
                prior = None
                cand_scores: list[dict[str, Any]] = []
                cands: list[dict[str, Any]] = []
                if not args.no_candidates:
                    try:
                        info = link_mention_with_candidates(
                            sp.mention,
                            sent_text,
                            languages=("zh", "en"),
                            min_score=min_ls,
                            entity_map_override=emap,
                            candidate_limit_per_lang=10,
                            rerank_topn=15,
                        )
                        chosen_qid = str(info.get("chosen_qid") or chosen_qid)
                        chosen_score = float(info.get("chosen_score") or chosen_score)
                        prior = info.get("override_hit")
                        cands = list(info.get("candidates") or [])
                        cand_scores = list(info.get("candidate_scores") or [])
                    except Exception:
                        # 候选检索失败时仍保留 span 的 object_qid
                        pass

                candidates_rows.append(
                    {
                        "source_id": source_id,
                        "source_url": url,
                        "source_label": label,
                        "citation_key": cite_key,
                        "seed_id": sd.seed_id,
                        "seed_qid": root_qid,
                        "sentence_idx": int(sp.sentence_idx),
                        "mention": sp.mention,
                        "candidates": cands,
                        "override_hit": prior,
                        "candidate_scores": cand_scores,
                    }
                )

                resolved_rows.append(
                    {
                        "source_id": source_id,
                        "source_url": url,
                        "source_label": label,
                        "citation_key": cite_key,
                        "seed_id": sd.seed_id,
                        "seed_qid": root_qid,
                        "sentence_idx": int(sp.sentence_idx),
                        "mention": sp.mention,
                        "ner_label": sp.ner_label,
                        "chosen_qid": chosen_qid if str(chosen_qid).startswith("Q") else "",
                        "scores": {"local": float(chosen_score), "global": 0.0, "total": float(chosen_score)},
                        "reasons": {
                            "local": {"link_score": float(chosen_score), "source": "rebuild_curated_ner_el"},
                            "global": {},
                            "prior": prior,
                        },
                    }
                )

            if mentions_rows:
                write_mentions_jsonl(root, mentions_rows)
                n_mentions += len(mentions_rows)
                by_seed_counts[sd.seed_id]["mentions"] += len(mentions_rows)
            if candidates_rows:
                write_candidates_jsonl(root, candidates_rows)
                n_candidates += len(candidates_rows)
                by_seed_counts[sd.seed_id]["candidates"] += len(candidates_rows)
            if resolved_rows:
                write_resolved_jsonl(root, resolved_rows)
                n_resolved += len(resolved_rows)
                by_seed_counts[sd.seed_id]["resolved"] += len(resolved_rows)

    print(f"完成：mentions={n_mentions} candidates={n_candidates} resolved={n_resolved}（已覆盖写入 {curated}）")
    for sid, cnt in sorted(by_seed_counts.items()):
        mm = int(cnt.get("mentions", 0))
        cc = int(cnt.get("candidates", 0))
        rr = int(cnt.get("resolved", 0))
        print(f"  seed_id={sid}: mentions={mm} candidates={cc} resolved={rr}")


if __name__ == "__main__":
    main()

