"""步骤三（补充）：基于模板与依存句法的关系抽取（与共现/实体链接区分）。

经典做法简述：
- 中文：传记常用触发词 + 正则槽位，再对槽位做实体链接；
- 英文：spaCy 依存（nsubj / prep / pobj）在含锚点句中抽取「主语—动词—介词宾语」结构。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..config import ROOT_ENTITY_QID
from ..graph_model import GraphBuild
from ..linking.entity_linking import link_mention_to_qid


@dataclass
class PatternRelation:
    sentence_idx: int
    object_qid: str
    object_mention: str
    predicate_label: str
    wikidata_prop_id: str
    snippet: str
    score: float
    method: str


def _zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / max(len(text), 1)


def _split_sentences(text: str, use_zh: bool) -> list[str]:
    text = text.replace("\r", "\n")
    if use_zh:
        parts = re.split(r"(?<=[。！？；\n])", text)
    else:
        parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 8]


def _sentence_has_anchor(s: str, anchors: list[str]) -> bool:
    return any(a in s.strip() for a in anchors if a)


def _clean_zh_slot(s: str) -> str:
    s = s.strip().strip("，,、")
    s = re.sub(r"[等及其].*$", "", s)
    return s.strip()[:80]


# (regex, predicate_label, Wikidata prop id hint, method tag)
_ZH_REGEX: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"毕业于([^。！？；\n]{2,50})"), "educated_at", "P69", "zh_template"),
    (re.compile(r"就读于([^。！？；\n]{2,50})"), "educated_at", "P69", "zh_template"),
    (re.compile(r"出生于([^。！？；\n]{2,50})"), "place_of_birth", "P19", "zh_template"),
    (re.compile(r"逝世于([^。！？；\n]{2,50})"), "place_of_death", "P20", "zh_template"),
    (re.compile(r"在([^。！？；\n]{2,40})工作"), "work_location", "P937", "zh_template"),
    (re.compile(r"任职于([^。！？；\n]{2,50})"), "employer", "P108", "zh_template"),
]


def _extract_zh_patterns(
    sentence: str,
    *,
    sentence_idx: int,
    entity_map: dict[str, dict],
    langs: tuple[str, ...],
    min_link_score: float,
    source_label: str,
) -> list[PatternRelation]:
    out: list[PatternRelation] = []
    seen: set[tuple[str, str]] = set()
    for pat, pred_label, pid, method in _ZH_REGEX:
        for m in pat.finditer(sentence):
            slot = _clean_zh_slot(m.group(1))
            if len(slot) < 2:
                continue
            qid, sc = link_mention_to_qid(
                slot,
                sentence,
                languages=langs,
                min_score=min_link_score,
                entity_map_override=entity_map,
            )
            if not qid or qid == ROOT_ENTITY_QID:
                continue
            key = (qid, pred_label)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                PatternRelation(
                    sentence_idx=sentence_idx,
                    object_qid=qid,
                    object_mention=slot[:120],
                    predicate_label=pred_label,
                    wikidata_prop_id=pid,
                    snippet=f"[{source_label}] {sentence[:280]} | zh_pattern={pred_label}",
                    score=sc,
                    method=method,
                )
            )
    return out


def _turing_head_token(doc: Any, anchors_en: list[str]) -> Any | None:
    """在英文句中找到指称图灵的 PERSON 或 'Turing' 词根。"""
    al = [a for a in anchors_en if len(a) >= 4]
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        et = ent.text.lower()
        if any(a.lower() in et for a in al) or ("turing" in et and "alan" in doc.text.lower()):
            return ent.root
    for t in doc:
        if t.lower_ == "turing":
            return t
    return None


def _pobj_entities(prep_token: Any) -> list[Any]:
    out = []
    for c in prep_token.children:
        if c.dep_ == "pobj":
            out.append(c)
    return out


def _extract_en_dep(
    sentence: str,
    sentence_idx: int,
    nlp: Any,
    anchors_en: list[str],
    *,
    entity_map: dict[str, dict],
    langs: tuple[str, ...],
    min_link_score: float,
    source_label: str,
) -> list[PatternRelation]:
    if not any(a in sentence for a in anchors_en):
        return []
    doc = nlp(sentence[:500_000])
    root = _turing_head_token(doc, anchors_en)
    if root is None:
        return []

    out: list[PatternRelation] = []
    seen: set[tuple[str, str]] = set()
    head = root.head

    # 被动：was born in London
    if head.lemma_.lower() in ("bear",) or head.text.lower() == "born":
        for c in head.children:
            if c.dep_ != "prep" or c.lower_ != "in":
                continue
            for pobj in _pobj_entities(c):
                if pobj.ent_type_ not in ("GPE", "LOC", "FAC", "ORG"):
                    continue
                mention = pobj.text.strip()
                qid, sc = link_mention_to_qid(
                    mention, sentence, languages=langs, min_score=min_link_score, entity_map_override=entity_map
                )
                if not qid or qid == ROOT_ENTITY_QID:
                    continue
                key = (qid, "place_of_birth")
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    PatternRelation(
                        sentence_idx=sentence_idx,
                        object_qid=qid,
                        object_mention=mention[:120],
                        predicate_label="place_of_birth",
                        wikidata_prop_id="P19",
                        snippet=f"[{source_label}] {sentence[:280]} | en_dep=born_in",
                        score=sc,
                        method="en_dependency",
                    )
                )

    # 主动：worked / studied / attended ... at / in ...
    if head.pos_ == "VERB":
        vlemma = head.lemma_.lower()
        for c in head.children:
            if c.dep_ != "prep" or c.lower_ not in ("at", "in", "for", "from"):
                continue
            for pobj in _pobj_entities(c):
                if pobj.ent_type_ not in ("ORG", "GPE", "FAC", "LOC"):
                    continue
                mention = pobj.text.strip()
                if vlemma in ("work", "serve", "join", "lecture", "write"):
                    pred_label, pid = "employer", "P108"
                elif vlemma in ("study", "attend", "read", "graduate"):
                    pred_label, pid = "educated_at", "P69"
                elif vlemma in ("live", "reside", "stay"):
                    pred_label, pid = "residence", "P551"
                elif vlemma in ("die",):
                    pred_label, pid = "place_of_death", "P20"
                else:
                    continue
                qid, sc = link_mention_to_qid(
                    mention, sentence, languages=langs, min_score=min_link_score, entity_map_override=entity_map
                )
                if not qid or qid == ROOT_ENTITY_QID:
                    continue
                key = (qid, pred_label)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    PatternRelation(
                        sentence_idx=sentence_idx,
                        object_qid=qid,
                        object_mention=mention[:120],
                        predicate_label=pred_label,
                        wikidata_prop_id=pid,
                        snippet=f"[{source_label}] {sentence[:280]} | en_dep={vlemma}_{c.lower_}",
                        score=sc,
                        method="en_dependency",
                    )
                )

    return out


def extract_pattern_relations(
    text: str,
    *,
    entity_map: dict[str, dict],
    root_anchors_zh: list[str],
    root_anchors_en: list[str],
    min_link_score: float,
    source_label: str,
) -> list[PatternRelation]:
    """仅在「含锚点」的句子上做关系模式（优先准确率，与共现窗口策略互补）。"""
    use_zh = _zh_ratio(text) >= 0.12
    anchors = root_anchors_zh if use_zh else root_anchors_en
    langs: tuple[str, ...] = ("zh", "en") if use_zh else ("en", "zh")
    sentences = _split_sentences(text, use_zh)
    out: list[PatternRelation] = []
    seen_global: set[tuple[str, str, str]] = set()

    if use_zh:
        for si, sent in enumerate(sentences):
            if not _sentence_has_anchor(sent, anchors):
                continue
            for pr in _extract_zh_patterns(
                sent,
                sentence_idx=si,
                entity_map=entity_map,
                langs=langs,
                min_link_score=min_link_score,
                source_label=source_label,
            ):
                gk = (pr.object_qid, pr.predicate_label, pr.object_mention[:40])
                if gk in seen_global:
                    continue
                seen_global.add(gk)
                out.append(pr)
        return out

    try:
        from .ner_link import _load_spacy

        nlp_en, _ = _load_spacy()
    except ImportError:
        nlp_en = None
    if nlp_en is None:
        return out

    for si, sent in enumerate(sentences):
        if not _sentence_has_anchor(sent, anchors):
            continue
        for pr in _extract_en_dep(
            sent,
            si,
            nlp_en,
            root_anchors_en,
            entity_map=entity_map,
            langs=langs,
            min_link_score=min_link_score,
            source_label=source_label,
        ):
            gk = (pr.object_qid, pr.predicate_label, pr.object_mention[:40])
            if gk in seen_global:
                continue
            seen_global.add(gk)
            out.append(pr)
    return out


def extract_pattern_relations_from_sentences(
    sentences: list[tuple[int, str]],
    *,
    entity_map: dict[str, dict],
    seed_anchors_zh: list[str],
    seed_anchors_en: list[str],
    min_link_score: float,
    source_label: str,
) -> list[PatternRelation]:
    """
    从「外部已选句子集合」直接抽取关系模式（不再要求句子含 anchor）。

    用途：配合 sentence routing，把“触发词归因句”也纳入关系抽取输入。
    仍复用现有中文模板 / 英文依存抽取与实体链接逻辑。
    """
    text = "\n".join([s.strip() for _i, s in sentences if s and s.strip()])
    if not text.strip():
        return []
    use_zh = _zh_ratio(text) >= 0.12
    anchors = seed_anchors_zh if use_zh else seed_anchors_en
    langs: tuple[str, ...] = ("zh", "en") if use_zh else ("en", "zh")
    out: list[PatternRelation] = []
    seen_global: set[tuple[str, str, str]] = set()

    if use_zh:
        for sent_idx, sent in [(i, s.strip()) for i, s in sentences if s and s.strip()]:
            for pr in _extract_zh_patterns(
                sent,
                sentence_idx=sent_idx,
                entity_map=entity_map,
                langs=langs,
                min_link_score=min_link_score,
                source_label=source_label,
            ):
                gk = (pr.object_qid, pr.predicate_label, pr.object_mention[:40])
                if gk in seen_global:
                    continue
                seen_global.add(gk)
                out.append(pr)
        return out

    try:
        from .ner_link import _load_spacy

        nlp_en, _ = _load_spacy()
    except ImportError:
        nlp_en = None
    if nlp_en is None:
        return out

    for sent_idx, sent in [(i, s.strip()) for i, s in sentences if s and s.strip()]:
        for pr in _extract_en_dep(
            sent,
            sent_idx,
            nlp_en,
            list(anchors),
            entity_map=entity_map,
            langs=langs,
            min_link_score=min_link_score,
            source_label=source_label,
        ):
            gk = (pr.object_qid, pr.predicate_label, pr.object_mention[:40])
            if gk in seen_global:
                continue
            seen_global.add(gk)
            out.append(pr)
    return out


def ingest_pattern_relations(
    g: GraphBuild,
    rels: list[PatternRelation],
    *,
    citation_key: str,
    source_url: str,
    root_qid: str = ROOT_ENTITY_QID,
) -> None:
    for pr in rels:
        g.ensure_node(root_qid)
        g.ensure_node(pr.object_qid, pr.object_mention[:120])
        g.add_edge(
            root_qid,
            pr.object_qid,
            pr.wikidata_prop_id,
            pr.predicate_label,
            "OUT",
            provenance="pattern_relation_extraction",
            citation_key=citation_key,
            snippet=f"{pr.snippet} | link={pr.score:.2f} method={pr.method}",
            source_url=source_url,
        )
