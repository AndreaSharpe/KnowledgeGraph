"""
从已落盘的 processed/curated JSONL 重构图与三元组，不调用 NER / 实体链接 API。

依赖：data/processed/sentences.jsonl、routing.jsonl、data/curated/resolved.jsonl
（由某次完整构建生成后保留；勿与 full 模式同一次清空逻辑冲突——本模式不调用 reset_stage_files）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .attribution.seed_config import load_attribution_config, load_seed_entities
from .extraction.ner_link import LinkedSpan, ingest_linked_spans, load_ner_link_config
from .extraction.relation_patterns import (
    extract_pattern_relations_from_sentences,
    ingest_pattern_relations,
)
from .graph_model import GraphBuild
from .io.sources_io import load_entity_map
from .structured.wikidata_layer import load_structured_graph_for_seeds


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_knowledge_graph_from_curated_stages(project_root: Path) -> tuple[GraphBuild, list[dict[str, str]]]:
    emap = load_entity_map(project_root / "sources" / "entity_map.csv")
    ncfg = load_ner_link_config(project_root)
    min_ls = float(ncfg.get("min_link_score", 0.14))
    seeds = load_seed_entities(project_root)
    load_attribution_config(project_root)

    seed_qids = [sd.qid for sd in seeds if sd.qid and sd.qid.startswith("Q")]
    g = load_structured_graph_for_seeds(seed_qids)

    processed = project_root / "data" / "processed"
    curated = project_root / "data" / "curated"
    sentences_rows = _read_jsonl(processed / "sentences.jsonl")
    routing_rows = _read_jsonl(processed / "routing.jsonl")
    resolved_rows = _read_jsonl(curated / "resolved.jsonl")

    if not sentences_rows or not resolved_rows:
        raise RuntimeError(
            "from_curated 模式需要非空的 data/processed/sentences.jsonl 与 data/curated/resolved.jsonl。"
            "请先运行一次 full 构建并保留这些文件，或检查路径。"
        )

    by_src_idx: dict[str, dict[int, str]] = defaultdict(dict)
    meta_by_source: dict[str, dict[str, str]] = {}
    for row in sentences_rows:
        sid = str(row.get("source_id") or "")
        si = int(row.get("sentence_idx", -1))
        if not sid or si < 0:
            continue
        by_src_idx[sid][si] = str(row.get("sentence") or "")
        if sid not in meta_by_source:
            meta_by_source[sid] = {
                "source_url": str(row.get("source_url") or ""),
                "source_label": str(row.get("source_label") or ""),
                "citation_key": str(row.get("citation_key") or ""),
            }

    routing_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in routing_rows:
        sid = str(row.get("source_id") or "")
        si = int(row.get("sentence_idx", -1))
        if sid and si >= 0:
            routing_by_key[(sid, si)] = row

    resolved_by_src_seed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolved_rows:
        sid = str(row.get("source_id") or "")
        see = str(row.get("seed_id") or "")
        if sid and see:
            resolved_by_src_seed[(sid, see)].append(row)

    seed_by_id = {s.seed_id: s for s in seeds}
    triple_rows: list[dict[str, str]] = []

    for (source_id, seed_id), res_list in sorted(resolved_by_src_seed.items()):
        sd = seed_by_id.get(seed_id)
        if not sd:
            continue
        root_qid = str(res_list[0].get("seed_qid") or sd.qid or "")
        if not root_qid.startswith("Q"):
            continue
        meta = meta_by_source.get(source_id, {})
        url = meta.get("source_url", "")
        cite_key = meta.get("citation_key", "")
        label = meta.get("source_label", source_id)
        idx_map = by_src_idx.get(source_id) or {}
        if not idx_map:
            continue
        max_i = max(idx_map.keys())
        sentences = [idx_map.get(i, "") for i in range(max_i + 1)]

        idx_set = {int(r.get("sentence_idx", -1)) for r in res_list if int(r.get("sentence_idx", -1)) >= 0}
        seed_items = [(i, sentences[i].strip()) for i in sorted(idx_set) if 0 <= i < len(sentences) and sentences[i].strip()]
        if not seed_items:
            continue

        derived_attr: dict[tuple[str, int], tuple[float, dict]] = {}
        for i, _s in seed_items:
            rr = routing_by_key.get((source_id, i))
            if not rr:
                derived_attr[(seed_id, i)] = (0.0, {"replayed_curated": True})
                continue
            best = None
            for a in rr.get("assigned") or []:
                if str(a.get("seed_id")) == seed_id:
                    best = (float(a.get("score") or 0.0), a.get("reasons") if isinstance(a.get("reasons"), dict) else {})
                    break
            derived_attr[(seed_id, i)] = best if best else (0.0, {"replayed_curated": True})

        spans: list[LinkedSpan] = []
        seen = set()
        for r in res_list:
            chosen = str(r.get("chosen_qid") or "").strip()
            mention = str(r.get("mention") or "").strip()
            si = int(r.get("sentence_idx", -1))
            if si < 0 or si >= len(sentences):
                continue
            ev = sentences[si].strip()
            ner = str(r.get("ner_label") or "")
            sc = 0.0
            sco = r.get("scores")
            if isinstance(sco, dict):
                sc = float(sco.get("total") or sco.get("local") or 0.0)
            k = (chosen, mention.lower(), si, ner)
            if k in seen:
                continue
            seen.add(k)
            if chosen == root_qid:
                continue
            spans.append(
                LinkedSpan(
                    sentence_idx=si,
                    object_qid=chosen,
                    mention=mention,
                    snippet=f"[{label}] {ev[:280]}",
                    score=sc,
                    ner_label=ner,
                    context="replayed_curated",
                )
            )

        ingest_linked_spans(g, spans, citation_key=cite_key, source_url=url, root_qid=root_qid)

        for sp in spans:
            ev = sp.snippet.split("] ", 1)[-1].strip() if sp.snippet else ""
            si = sp.sentence_idx
            sc, rs = derived_attr.get((seed_id, si), (0.0, {"unattributed": True}))
            rel = f"{float(sc):.4f}" if sc else ""
            reasons = json.dumps(rs, ensure_ascii=False)[:800] if rs else ""
            triple_rows.append(
                {
                    "subject_qid": root_qid,
                    "seed_id": sd.seed_id,
                    "seed_qid": root_qid,
                    "sentence_idx": str(si),
                    "relevance_score": rel,
                    "reasons": reasons,
                    "evidence_sentence": ev[:500],
                    "predicate": "cooccurrence_linked",
                    "predicate_prop_id": "",
                    "object_mention": sp.mention,
                    "object_qid": sp.object_qid,
                    "object_is_literal": "1" if not sp.object_qid else "0",
                    "ner_label": sp.ner_label,
                    "link_score": f"{sp.score:.4f}",
                    "citation_key": cite_key,
                    "source_url": url,
                    "snippet": sp.snippet,
                    "extraction_method": f"ner_entity_linking:{sp.context}",
                }
            )

        try:
            prels = extract_pattern_relations_from_sentences(
                seed_items,
                entity_map=emap,
                seed_anchors_zh=list(sd.anchors_zh),
                seed_anchors_en=list(sd.anchors_en),
                min_link_score=min_ls,
                source_label=label,
            )
        except ImportError:
            prels = []
        prels = [pr for pr in prels if pr.object_qid != root_qid]
        ingest_pattern_relations(g, prels, citation_key=cite_key, source_url=url, root_qid=root_qid)
        for pr in prels:
            ev = pr.snippet.split("] ", 1)[-1].strip() if pr.snippet else ""
            si = pr.sentence_idx
            sc, rs = derived_attr.get((seed_id, si), (0.0, {"unattributed": True}))
            rel = f"{float(sc):.4f}" if sc else ""
            reasons = json.dumps(rs, ensure_ascii=False)[:800] if rs else ""
            triple_rows.append(
                {
                    "subject_qid": root_qid,
                    "seed_id": sd.seed_id,
                    "seed_qid": root_qid,
                    "sentence_idx": str(si),
                    "relevance_score": rel,
                    "reasons": reasons,
                    "evidence_sentence": ev[:500],
                    "predicate": pr.predicate_label,
                    "predicate_prop_id": pr.wikidata_prop_id,
                    "object_mention": pr.object_mention,
                    "object_qid": pr.object_qid,
                    "object_is_literal": "0",
                    "ner_label": "",
                    "link_score": f"{pr.score:.4f}",
                    "citation_key": cite_key,
                    "source_url": url,
                    "snippet": pr.snippet,
                    "extraction_method": pr.method,
                }
            )

    from .attribution.triple_merge import merge_triple_rows

    triple_rows = merge_triple_rows(triple_rows)
    return g, triple_rows


def build_export_only_from_disk(project_root: Path) -> tuple[GraphBuild, list[dict[str, str]]]:
    """仅从 data/*.csv 加载图与三元组，用于快速再导出（+MIL）。"""
    from .io.export_io import read_triples_csv
    from .io.graph_csv_io import load_graph_build_from_data_csv

    data_dir = project_root / "data"
    g = load_graph_build_from_data_csv(data_dir)
    triple_rows = read_triples_csv(data_dir / "triples_extracted.csv")
    return g, triple_rows
