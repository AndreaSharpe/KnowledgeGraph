"""
知识图谱构建（教科书式流水线）

1. 结构化知识：Wikidata API（中心实体陈述）
2. 文本采集：维基摘要、book_excerpt / articles、PDF 节选
3. 信息抽取：锚点窗口内 spaCy NER → 实体链接；含锚点句上的中文模板 / 英文依存关系抽取 → 写入图
4. 导出：nodes.csv、relationships.csv、graph_summary.json、triples_extracted.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from .attribution.build_mode import load_build_mode
from .build_replay import build_export_only_from_disk, build_knowledge_graph_from_curated_stages
from .attribution.seed_config import (
    load_attribution_config,
    load_entity_linking_config,
    load_extraction_profile,
    load_seed_entities,
    pick_ner_backend_for_source,
)
from .attribution.seed_router import route_sentences
from .attribution.audit_export import write_sentence_attribution_jsonl
from .attribution.triple_merge import merge_triple_rows
from .config import ROOT_ENTITY_QID
from .extraction.ner_link import (
    LinkedSpan,
    extract_linked_spans_from_sentences,
    ingest_linked_spans,
    load_ner_link_config,
)
from .extraction.relation_patterns import (
    extract_pattern_relations_from_sentences,
    ingest_pattern_relations,
)
from .extraction.event_patterns import extract_events_from_sentences, ingest_events
from .graph_model import GraphBuild
from .io.export_io import write_graph_csv_json, write_triples_csv
from .relation.mil_ingest import apply_mil_to_export_if_present
from .io.stage_io import (
    reset_stage_files,
    write_candidates_jsonl,
    write_mentions_jsonl,
    write_processed_sentences_jsonl,
    write_resolved_jsonl,
    write_routing_jsonl,
)
from .io.sources_io import load_bibliography, load_entity_kind_by_qid, load_entity_map
from .linking.collective_linking import (
    Candidate,
    CollectiveConfig,
    MentionNode,
    collective_decode_window,
    fetch_neighbor_sets,
)
from .linking.entity_linking import link_mention_to_qid, link_mention_with_candidates
from .structured.wikidata_layer import load_structured_graph_for_seeds
from .structured.wikidata_api import wbgetentities, pick_label


def build_knowledge_graph(project_root: Path, *, mode: str | None = None) -> tuple[GraphBuild, list[dict[str, str]]]:
    """执行步骤 1–3，返回图与用于 triples CSV 的行列表。

    构建模式由 ``sources/build_config.json`` 的 ``mode`` 字段控制，也可用参数 ``mode`` 覆盖（如 full / export_only / from_curated）。
    - ``full``：默认，清空中间层并跑 NER/链接（原行为）
    - ``export_only``：只读 ``data/*.csv`` 再导出（适合已有导出、只合并 MIL 等）
    - ``from_curated``：用 ``sentences/routing/resolved`` JSONL 重放，不调用 NER/链接
    """
    mode = (mode or load_build_mode(project_root)).strip().lower()
    if mode not in ("full", "export_only", "from_curated"):
        mode = "full"
    if mode == "export_only":
        return build_export_only_from_disk(project_root)
    if mode == "from_curated":
        return build_knowledge_graph_from_curated_stages(project_root)

    emap_path = project_root / "sources" / "entity_map.csv"
    emap = load_entity_map(emap_path)
    kind_by_qid = load_entity_kind_by_qid(emap_path)
    ncfg = load_ner_link_config(project_root)
    min_ls = float(ncfg.get("min_link_score", 0.14))

    seeds = load_seed_entities(project_root)
    at_cfg = load_attribution_config(project_root)
    ex_profile = load_extraction_profile(project_root)
    el_cfg = load_entity_linking_config(project_root)

    # --- 步骤 1：Wikidata 结构化层（多 seed 合并）---
    seed_qids = [sd.qid for sd in seeds if sd.qid and sd.qid.startswith("Q")]
    g = load_structured_graph_for_seeds(seed_qids)
    # 给 seed 节点加上类型标签（Neo4j :LABEL）
    for sd in seeds:
        if not sd.qid or not sd.qid.startswith("Q"):
            continue
        st = (sd.type or "").strip()
        if st:
            g.ensure_node(sd.qid, labels=(st,))

    # --- 步骤 2–3：文本 + NER + 实体链接 + 模式/依存关系 ---
    from .extraction.ner_link import _split_sentences, _zh_ratio
    from .sources.text_sources import collect_text_sources

    triple_rows: list[dict[str, str]] = []
    audit_path = project_root / "data" / "sentence_attribution.jsonl"
    # 每次构建覆盖审计文件（避免无限增长）
    try:
        audit_path.write_text("", encoding="utf-8")
    except Exception:
        pass
    reset_stage_files(project_root)

    for ch, _prov, cite_key, url in collect_text_sources(project_root):
        label = f"{ch.lang}:{ch.title}"
        source_id = url or label

        use_zh = _zh_ratio(ch.text) >= 0.12
        sentences = _split_sentences(ch.text, use_zh)
        try:
            write_processed_sentences_jsonl(
                project_root,
                source_id=source_id,
                source_url=url,
                source_label=label,
                citation_key=cite_key,
                sentences=sentences,
            )
        except Exception:
            pass

        routed = route_sentences(sentences, seeds=seeds, cfg=at_cfg)
        sent_attr_by_seed: dict[tuple[str, int], tuple[float, dict]] = {}
        for i, ra in enumerate(routed):
            for a in ra.assigned:
                sent_attr_by_seed[(a.seed_id, i)] = (float(a.score), a.reasons or {})

        idx_by_seed: dict[str, set[int]] = {}
        for i, ra in enumerate(routed):
            for a in ra.assigned:
                idx_by_seed.setdefault(a.seed_id, set()).add(i)

        if not idx_by_seed:
            for sd in seeds:
                if sd.seed_id == "turing_person":
                    idx_by_seed[sd.seed_id] = set(range(len(sentences)))
                    break

        if at_cfg.window_sentences and at_cfg.window_sentences > 0:
            w = int(at_cfg.window_sentences)
            for sid, sidx in list(idx_by_seed.items()):
                expanded: set[int] = set(sidx)
                for i in list(sidx):
                    for j in range(max(0, i - w), min(len(sentences), i + w + 1)):
                        expanded.add(j)
                idx_by_seed[sid] = expanded

        # 对窗口扩展出来但未被路由直接归因的句子，生成“派生归因”
        # 规则：取 seed 在窗口内最近/最高分的已归因句作为来源；保证每个 (seed, sentence_idx) 有确定 score/reasons。
        derived_attr: dict[tuple[str, int], tuple[float, dict]] = dict(sent_attr_by_seed)
        if at_cfg.window_sentences and at_cfg.window_sentences > 0:
            w = int(at_cfg.window_sentences)
            for sd in seeds:
                base_idxs = sorted({i for (sid, i) in sent_attr_by_seed.keys() if sid == sd.seed_id})
                if not base_idxs:
                    continue
                for i in sorted(idx_by_seed.get(sd.seed_id, set())):
                    if (sd.seed_id, i) in derived_attr:
                        continue
                    # 找邻近 base idx
                    cand = [bi for bi in base_idxs if abs(bi - i) <= w]
                    if not cand:
                        continue
                    best_bi = max(cand, key=lambda bi: sent_attr_by_seed.get((sd.seed_id, bi), (0.0, {}))[0])
                    sc, rs = sent_attr_by_seed.get((sd.seed_id, best_bi), (0.0, {}))
                    r2 = dict(rs or {})
                    r2["window_expanded"] = True
                    r2["window_from_sentence_idx"] = best_bi
                    derived_attr[(sd.seed_id, i)] = (float(sc), r2)

        backend = pick_ner_backend_for_source(ex_profile, url)
        crf_model_path = ex_profile.crf_model_path

        # 归因审计落盘（逐句）
        try:
            audit_rows = []
            for i, ra in enumerate(routed):
                assigned = []
                for a in ra.assigned:
                    assigned.append(
                        {
                            "seed_id": a.seed_id,
                            "seed_qid": a.seed_qid,
                            "score": float(a.score),
                            "reasons": a.reasons,
                        }
                    )
                audit_rows.append({"sentence_idx": i, "sentence": ra.sentence, "assigned": assigned})
            write_sentence_attribution_jsonl(audit_path, source_url=url, source_label=label, rows=audit_rows)
            write_routing_jsonl(
                project_root,
                source_id=source_id,
                source_url=url,
                source_label=label,
                citation_key=cite_key,
                routed_rows=audit_rows,
            )
        except Exception:
            pass

        for sd in seeds:
            if sd.seed_id not in idx_by_seed:
                continue
            sent_idxs = sorted(idx_by_seed[sd.seed_id])
            seed_items = [(i, sentences[i]) for i in sent_idxs if 0 <= i < len(sentences)]
            seed_sents = [s for _i, s in seed_items]
            if not seed_sents:
                continue

            root_qid = sd.qid or ""
            if not root_qid:
                guess = (sd.anchors_zh[0] if sd.anchors_zh else (sd.anchors_en[0] if sd.anchors_en else "")).strip()
                row = emap.get(guess) or emap.get(guess.lower())
                if row and row.get("wikidata_id"):
                    root_qid = row["wikidata_id"]
            if not root_qid:
                continue

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
                    from pathlib import Path as _Path

                    mp = (
                        (_Path(project_root) / str(crf_model_path)).resolve()
                        if crf_model_path
                        else (_Path(project_root) / "models/crf_ner.pkl").resolve()
                    )
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
                        from .extraction.crf_ner import extract_entities_with_crf_zh

                        ents = extract_entities_with_crf_zh(seed_sents, model_path=mp)
                        spans = []
                        seen = set()
                        for e in ents:
                            men = e.mention.strip()
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
                            # 将 evidence_sentence 映射回 sentence_idx（确定性：用 seed_items 的 idx 做精确匹配）
                            sent_idx = -1
                            for ii, ss in seed_items:
                                if ss.strip() == e.evidence_sentence.strip():
                                    sent_idx = ii
                                    break
                            spans.append(
                                LinkedSpan(
                                    sentence_idx=sent_idx,
                                    object_qid=obj_qid,
                                    mention=men,
                                    snippet=f"[{label}] {e.evidence_sentence[:280]}",
                                    score=float(sc),
                                    ner_label=e.label,
                                    context="crf",
                                )
                            )
            except ImportError as e:
                warnings.warn(str(e), stacklevel=1)
                spans = []

            spans = [sp for sp in spans if sp.object_qid != root_qid]

            # --- 事件抽取（Trigger→Arguments；仅在 routed seed_items 上运行）---
            try:
                linked_by_sentence: dict[int, list[tuple[str, str, str]]] = {}
                for sp in spans:
                    si = int(sp.sentence_idx)
                    linked_by_sentence.setdefault(si, []).append(
                        (str(sp.object_qid or ""), str(sp.mention or ""), str(sp.ner_label or ""))
                    )
                events = extract_events_from_sentences(
                    seed_items,
                    seed_id=sd.seed_id,
                    seed_qid=root_qid,
                    source_id=source_id,
                    source_url=url,
                    source_label=label,
                    citation_key=cite_key,
                    entity_map=emap,
                    linked_by_sentence=linked_by_sentence,
                    min_link_score=min_ls,
                )
                if events:
                    ingest_events(g, events)
            except Exception:
                # 事件抽取失败不应阻断主流程
                pass

            # --- Phase 1 中间层：mentions / candidates / resolved 落盘（基于当前实现，不改抽取门控逻辑）---
            try:
                mentions_rows = []
                candidates_rows = []
                resolved_rows = []
                mention_nodes: list[MentionNode] = []
                all_candidate_qids: list[str] = []
                for sp in spans:
                    ev = sp.snippet.split("] ", 1)[-1].strip() if sp.snippet else ""
                    mentions_rows.append(
                        {
                            "source_id": source_id,
                            "source_url": url,
                            "source_label": label,
                            "citation_key": cite_key,
                            "seed_id": sd.seed_id,
                            "seed_qid": root_qid,
                            "sentence_idx": int(sp.sentence_idx),
                            "sentence": ev,
                            "mention": sp.mention,
                            "ner_label": sp.ner_label,
                            "ner_backend": backend,
                            "mention_context": sp.context,
                            "char_start": None,
                            "char_end": None,
                        }
                    )

                    # literal 或空 qid：仍然记录 mention，并写 resolved（chosen_qid 为空），但不做候选检索/协同。
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
                                    "local": {
                                        "link_score": float(sp.score),
                                        "source": "literal_no_wikidata_link",
                                    },
                                    "global": {},
                                    "prior": None,
                                },
                            }
                        )
                        continue

                    langs = ("zh", "en") if use_zh else ("en", "zh")
                    info = link_mention_with_candidates(
                        sp.mention,
                        ev,
                        languages=langs,
                        min_score=min_ls,
                        entity_map_override=emap,
                    )
                    cand_scores = list(info.get("candidate_scores") or [])
                    top = cand_scores[: max(1, int(el_cfg.collective_top_k_candidates))]
                    cands: list[Candidate] = []
                    for cs in top:
                        q = str(cs.get("qid") or "")
                        if not q.startswith("Q"):
                            continue
                        cands.append(
                            Candidate(
                                qid=q,
                                score=float(cs.get("score") or 0.0),
                                meta=dict(cs.get("breakdown") or {}),
                            )
                        )
                        all_candidate_qids.append(q)
                    node_key = f"{source_id}|{sd.seed_id}|{int(sp.sentence_idx)}|{sp.mention}|{sp.ner_label}"
                    if cands:
                        mention_nodes.append(
                            MentionNode(
                                key=node_key,
                                sentence_idx=int(sp.sentence_idx),
                                mention=sp.mention,
                                candidates=tuple(cands),
                                chosen_qid=sp.object_qid,
                            )
                        )
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
                            "candidates": info.get("candidates", []),
                            "override_hit": info.get("override_hit"),
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
                            "chosen_qid": sp.object_qid,
                            "scores": {"local": float(sp.score), "global": 0.0, "total": float(sp.score)},
                            "reasons": {
                                "local": {
                                    "link_score": float(sp.score),
                                    "source": "existing_pipeline",
                                    "breakdown": info.get("local_breakdown") if isinstance(info, dict) else None,
                                },
                                "global": {},
                                "prior": info.get("override_hit") if isinstance(info, dict) else None,
                            },
                        }
                    )

                # --- 协同链接（Collective EL）---
                if (
                    getattr(el_cfg, "collective_enabled", False)
                    and mention_nodes
                    and float(getattr(el_cfg, "collective_lambda_coherence", 0.0)) > 0
                ):
                    cfg = CollectiveConfig(
                        enabled=True,
                        window_sentences=int(getattr(el_cfg, "collective_window_sentences", 2)),
                        top_k_candidates=int(getattr(el_cfg, "collective_top_k_candidates", 6)),
                        lambda_coherence=float(getattr(el_cfg, "collective_lambda_coherence", 0.35)),
                        coherence_props=tuple(getattr(el_cfg, "collective_coherence_props", ())),
                        max_entities_to_fetch=int(getattr(el_cfg, "collective_max_entities_to_fetch", 120)),
                    )
                    neighbor_sets = fetch_neighbor_sets(
                        all_candidate_qids,
                        props=cfg.coherence_props,
                        languages=("zh|en" if use_zh else "en|zh"),
                        max_entities=int(cfg.max_entities_to_fetch),
                    )
                    decoded: dict[str, dict[str, Any]] = {}
                    centers = sorted({n.sentence_idx for n in mention_nodes})
                    for center in centers:
                        win = [n for n in mention_nodes if abs(int(n.sentence_idx) - int(center)) <= int(cfg.window_sentences)]
                        if len(win) < 2:
                            continue
                        decoded.update(
                            collective_decode_window(
                                win,
                                neighbor_sets=neighbor_sets,
                                lam=float(cfg.lambda_coherence),
                                max_iters=3,
                            )
                        )

                    if decoded:
                        # resolved_rows：写 global 贡献 + before/after
                        for rr in resolved_rows:
                            k = f"{rr.get('source_id')}|{rr.get('seed_id')}|{int(rr.get('sentence_idx'))}|{rr.get('mention')}|{rr.get('ner_label')}"
                            if k not in decoded:
                                continue
                            d = decoded[k]
                            rr.setdefault("scores", {})
                            rr["scores"]["global"] = float(d.get("global_score") or 0.0)
                            rr["scores"]["total"] = float(d.get("total_score") or rr["scores"].get("local", 0.0))
                            rr.setdefault("reasons", {})
                            rr["reasons"]["global"] = {
                                "lambda_coherence": float(cfg.lambda_coherence),
                                "window_sentences": int(cfg.window_sentences),
                                "coherence_props": list(cfg.coherence_props),
                                "changed_from": d.get("changed_from"),
                            }
                            if d.get("changed_from"):
                                rr["reasons"]["global"]["changed_to"] = d.get("chosen_qid")
                                rr["chosen_qid"] = d.get("chosen_qid")

                        # spans：回写最终 chosen_qid（影响入图与 triples）
                        for sp in spans:
                            k = f"{source_id}|{sd.seed_id}|{int(sp.sentence_idx)}|{sp.mention}|{sp.ner_label}"
                            if k not in decoded:
                                continue
                            d = decoded[k]
                            new_qid = d.get("chosen_qid")
                            if isinstance(new_qid, str) and new_qid.startswith("Q") and new_qid != sp.object_qid:
                                sp.object_qid = new_qid
                                sp.score = float(d.get("total_score") or sp.score)

                if mentions_rows:
                    write_mentions_jsonl(project_root, mentions_rows)
                if candidates_rows:
                    write_candidates_jsonl(project_root, candidates_rows)
                if resolved_rows:
                    write_resolved_jsonl(project_root, resolved_rows)
            except Exception:
                pass

            # 不将 NER/EL 共现证据边写入 GraphBuild。
            # spans/mentions/candidates/resolved 仍用于后续 DS/RE 与审计。
            # ingest_linked_spans(g, spans, citation_key=cite_key, source_url=url, root_qid=root_qid)

            for sp in spans:
                # 直接使用 sentence_idx 从派生归因表取 score/reasons
                ev = sp.snippet.split("] ", 1)[-1].strip() if sp.snippet else ""
                si = sp.sentence_idx
                sc, rs = derived_attr.get((sd.seed_id, si), (0.0, {"unattributed": True}))
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
            # 同上：pattern 规则边属于证据层诊断，不写入最终 GraphBuild。
            # ingest_pattern_relations(g, prels, citation_key=cite_key, source_url=url, root_qid=root_qid)
            for pr in prels:
                ev = pr.snippet.split("] ", 1)[-1].strip() if pr.snippet else ""
                si = pr.sentence_idx
                sc, rs = derived_attr.get((sd.seed_id, si), (0.0, {"unattributed": True}))
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

    triple_rows = merge_triple_rows(triple_rows)

    # 经典 KG：用离线 entity_map 的 kind 给节点补 :LABEL（不要求覆盖所有实体，但要可逐步扩充）
    for qid, kind in kind_by_qid.items():
        if qid in g.nodes and kind:
            g.ensure_node(qid, labels=(kind,))

    # 经典 KG：确保节点 name 是实体值而非 QID（中文优先，无则英文）
    try:
        qids = [nid for nid, n in g.nodes.items() if isinstance(nid, str) and nid.startswith("Q") and (str(n.get("name") or "") == nid or not str(n.get("name") or "").strip())]
        if qids:
            ents = wbgetentities(sorted(set(qids)), props="labels", languages="zh-hans|zh|en")
            for qid in qids:
                ent = ents.get(qid) if isinstance(ents, dict) else None
                if isinstance(ent, dict):
                    lab = pick_label(ent)
                    if lab and lab != qid:
                        g.ensure_node(qid, name=lab)
    except Exception:
        pass

    # --- 结构化类型补齐：为缺少细粒度类型的 Q 节点补 Person/Award/... ---
    try:
        from .structured.wikidata_layer import enrich_kind_labels_for_graph

        enrich_kind_labels_for_graph(g)
    except Exception:
        pass
    return g, triple_rows


def export_all(
    project_root: Path,
    g: GraphBuild,
    triple_rows: list[dict[str, str]],
) -> None:
    """步骤 4：导出到 data/。

    若存在 ``data/triples_mil_extracted.csv`` 与/或 ``data/curated/re_predictions*.jsonl``，
    则合并三元组并将通过阈值的 MIL 边写入图（与同 subject/object/prop 的已有边去重）。
    """
    data_dir = project_root / "data"
    bib = load_bibliography(project_root / "sources" / "bibliography.json")
    triple_rows, _n_mil_edges = apply_mil_to_export_if_present(project_root, g, triple_rows)
    write_graph_csv_json(g, data_dir, bibliography=bib)
    write_triples_csv(data_dir / "triples_extracted.csv", triple_rows)
