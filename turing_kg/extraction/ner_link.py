"""步骤三：命名实体识别（spaCy）+ 实体链接（entity_linking）。"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ROOT_ENTITY_QID
from ..graph_model import GraphBuild
from ..linking.entity_linking import link_mention_to_qid


@dataclass
class LinkedSpan:
    sentence_idx: int
    object_qid: str
    mention: str
    snippet: str
    score: float
    ner_label: str
    context: str = "anchor"


def _normalize_mention(m: str) -> str:
    s = (m or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'“”‘’()（）[]【】")
    return s


def _map_to_seven_labels(spacy_label: str) -> str | None:
    """
    将 spaCy 标签映射到项目要求的七类（PER/ORG/LOC/TIME/DATE/MONEY/PERCENT）。
    未覆盖的类别返回 None（直接丢弃）。
    """
    lab = (spacy_label or "").strip().upper()
    if lab == "PERSON":
        return "PER"
    if lab == "ORG":
        return "ORG"
    if lab in ("GPE", "LOC", "FAC"):
        return "LOC"
    if lab == "DATE":
        return "DATE"
    if lab == "TIME":
        return "TIME"
    if lab == "MONEY":
        return "MONEY"
    if lab in ("PERCENT",):
        return "PERCENT"
    return None


_NUMERIC_LABELS = {"DATE", "TIME", "MONEY", "PERCENT"}


_RE_DATE = re.compile(r"((?:19|20)\d{2}年\d{1,2}月\d{1,2}[日号]|(?:19|20)\d{2}年\d{1,2}月|(?:19|20)\d{2}年)")
_RE_TIME = re.compile(r"(\d{1,2}[时點点]\d{1,2}分?|\d{1,2}[时點点]|\d{1,2}:\d{2})")
_RE_PERCENT = re.compile(r"(\d+(?:\.\d+)?\s*%)")
_RE_MONEY = re.compile(r"([¥￥]\s*\d+(?:\.\d+)?|£\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:元|人民币|美元|英镑))")


def _extract_numeric_mentions(sent: str) -> list[tuple[str, str]]:
    """
    用规则补强数值类实体（DATE/TIME/MONEY/PERCENT）。
    返回 (mention, label) 其中 label 为七类之一。
    """
    out: list[tuple[str, str]] = []
    for m in _RE_DATE.finditer(sent):
        out.append((m.group(1), "DATE"))
    for m in _RE_TIME.finditer(sent):
        out.append((m.group(1), "TIME"))
    for m in _RE_PERCENT.finditer(sent):
        out.append((m.group(1), "PERCENT"))
    for m in _RE_MONEY.finditer(sent):
        out.append((m.group(1), "MONEY"))
    # 去重保持顺序
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for men, lab in out:
        key = (_normalize_mention(men), lab)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        uniq.append((key[0], lab))
    return uniq


# spaCy 中文小模型对地名/国家（GPE/LOC）召回可能偏低；用极轻量规则兜底一批高价值地理 mention。
# 这些 mention 对 DS 的 P19/P20/P27/P17 等关系至关重要。
_EXTRA_GEO_MENTIONS_ZH: tuple[str, ...] = (
    "英国",
    "伦敦",
    "英格兰",
    "苏格兰",
    "威尔士",
    "爱尔兰",
    "曼彻斯特",
    "普林斯顿",
    "剑桥",
    "帕丁顿",
    "布莱切利",
)


def _extract_geo_mentions_zh(sent: str) -> list[tuple[str, str]]:
    """返回 (mention, label)；label 固定为 LOC。"""
    out: list[tuple[str, str]] = []
    s = sent or ""
    for men in _EXTRA_GEO_MENTIONS_ZH:
        if men and men in s:
            out.append((men, "LOC"))
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for men, lab in out:
        men2 = _normalize_mention(men)
        if not men2 or men2 in seen:
            continue
        seen.add(men2)
        uniq.append((men2, lab))
    return uniq


def load_ner_link_config(project_root: Path) -> dict[str, Any]:
    p = project_root / "sources" / "ner_link_config.json"
    if not p.is_file():
        return {
            "root_anchors_zh": ["艾伦·图灵", "阿兰·图灵", "图灵"],
            "root_anchors_en": ["Alan Turing", "Alan Mathison Turing", "Turing"],
            "min_link_score": 0.14,
            "anchor_window_sentences": 1,
        }
    return json.loads(p.read_text(encoding="utf-8"))


_nlp_en = None
_nlp_zh = None


def _load_spacy():
    global _nlp_en, _nlp_zh
    try:
        import spacy
    except ImportError as e:
        raise ImportError("请安装：pip install spacy && python -m spacy download en_core_web_sm zh_core_web_sm") from e
    if _nlp_en is None:
        try:
            _nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            _nlp_en = None
            warnings.warn("缺少 en_core_web_sm", stacklevel=1)
    if _nlp_zh is None:
        try:
            _nlp_zh = spacy.load("zh_core_web_sm")
        except OSError:
            _nlp_zh = None
            warnings.warn("缺少 zh_core_web_sm", stacklevel=1)
    return _nlp_en, _nlp_zh


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


def _expand_anchor_indices(n: int, anchor_idxs: set[int], window: int) -> set[int]:
    """以含锚点句为中心，向左右各扩展 window 条句子（教科书式局部上下文）。"""
    if window <= 0:
        return set(anchor_idxs)
    out: set[int] = set()
    for i in anchor_idxs:
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        for j in range(lo, hi):
            out.add(j)
    return out


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def extract_linked_spans(
    text: str,
    *,
    entity_map: dict[str, dict],
    root_anchors_zh: list[str],
    root_anchors_en: list[str],
    min_link_score: float,
    source_label: str,
    anchor_window_sentences: int = 1,
) -> list[LinkedSpan]:
    use_zh = _zh_ratio(text) >= 0.12
    anchors = root_anchors_zh if use_zh else root_anchors_en
    nlp_en, nlp_zh = _load_spacy()
    nlp = nlp_zh if use_zh else nlp_en
    if nlp is None:
        nlp = nlp_en or nlp_zh
    if nlp is None:
        return []

    langs = ("zh", "en") if use_zh else ("en", "zh")
    sentences = _split_sentences(text, use_zh)
    anchor_idxs = {i for i, s in enumerate(sentences) if _sentence_has_anchor(s, anchors)}
    eligible = _expand_anchor_indices(len(sentences), anchor_idxs, max(0, anchor_window_sentences))
    out: list[LinkedSpan] = []
    seen_pair: set[tuple[str, str]] = set()

    for si, sent in enumerate(sentences):
        if si not in eligible:
            continue
        in_anchor = si in anchor_idxs
        ctx = "anchor" if in_anchor else "neighbor_window"
        doc = nlp(sent[:1_000_000])
        root_spans: list[tuple[int, int]] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            et = ent.text.strip()
            if any(a in et or et in a for a in anchors if len(a) >= 2):
                root_spans.append((ent.start_char, ent.end_char))

        for ent in doc.ents:
            mapped = _map_to_seven_labels(ent.label_)
            if mapped is None:
                continue
            if not in_anchor and ent.label_ == "PERSON":
                continue
            span = (ent.start_char, ent.end_char)
            if any(_overlap(span, rs) for rs in root_spans) and ent.label_ == "PERSON":
                continue
            mention = _normalize_mention(ent.text)
            if len(mention) < 2:
                continue
            # 数值类实体默认不做 Wikidata 链接：作为 literal 导出（避免“11分钟/24岁”误链到影片/法律等）。
            if mapped in _NUMERIC_LABELS:
                key = ("", f"{mapped}:{mention}".lower())
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append(
                    LinkedSpan(
                        sentence_idx=si,
                        object_qid="",
                        mention=mention,
                        snippet=f"[{source_label}] {sent[:280]}",
                        score=1.0,
                        ner_label=mapped,
                        context=ctx,
                    )
                )
            else:
                qid, sc = link_mention_to_qid(
                    mention,
                    sent,
                    languages=langs,
                    min_score=min_link_score,
                    entity_map_override=entity_map,
                )
                if not qid or qid == ROOT_ENTITY_QID:
                    continue
                key = (qid, mention.lower())
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append(
                    LinkedSpan(
                        sentence_idx=si,
                        object_qid=qid,
                        mention=mention,
                        snippet=f"[{source_label}] {sent[:280]}",
                        score=sc,
                        ner_label=mapped,
                        context=ctx,
                    )
                )

        # 规则补强：数值类实体一般不做 Wikidata 链接（多为字面值），但仍导出可审计。
        # 这里以 object_qid 留空的方式输出，后续导出增强时可单列 literal_value。
        for men, lab in _extract_numeric_mentions(sent):
            key = ("", f"{lab}:{men}".lower())
            if key in seen_pair:
                continue
            seen_pair.add(key)
            out.append(
                LinkedSpan(
                    sentence_idx=si,
                    object_qid="",
                    mention=men,
                    snippet=f"[{source_label}] {sent[:280]}",
                    score=1.0,
                    ner_label=lab,
                    context=ctx,
                )
            )

        # 规则兜底：补齐高价值 LOC（仅中文）；仍走 EL 获取 QID，便于 DS 匹配 Wikidata 正例。
        if use_zh:
            for men, lab in _extract_geo_mentions_zh(sent):
                qid, sc = link_mention_to_qid(
                    men,
                    sent,
                    languages=langs,
                    min_score=min_link_score,
                    entity_map_override=entity_map,
                )
                if not qid or qid == ROOT_ENTITY_QID:
                    continue
                key = (qid, men.lower())
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append(
                    LinkedSpan(
                        sentence_idx=si,
                        object_qid=qid,
                        mention=men,
                        snippet=f"[{source_label}] {sent[:280]}",
                        score=sc,
                        ner_label=lab,
                        context="geo_rule",
                    )
                )
    return out


def extract_linked_spans_from_sentences(
    sentences: list[tuple[int, str]],
    *,
    entity_map: dict[str, dict],
    seed_anchors_zh: list[str],
    seed_anchors_en: list[str],
    min_link_score: float,
    source_label: str,
) -> list[LinkedSpan]:
    """
    从「外部已选句子集合」直接做 spaCy NER + 实体链接。

    用途：
    - 句子路由/触发词归因后，句子本身可能不含 anchor；此时不能再用“含锚点句筛选”作为前置条件。
    - 仍复用现有的 spaCy 模型、allowed label、PERSON 降噪逻辑与实体链接函数。
    """
    text = "\n".join([s.strip() for _i, s in sentences if s and s.strip()])
    if not text.strip():
        return []
    use_zh = _zh_ratio(text) >= 0.12
    anchors = seed_anchors_zh if use_zh else seed_anchors_en
    nlp_en, nlp_zh = _load_spacy()
    nlp = nlp_zh if use_zh else nlp_en
    if nlp is None:
        nlp = nlp_en or nlp_zh
    if nlp is None:
        return []

    langs = ("zh", "en") if use_zh else ("en", "zh")
    out: list[LinkedSpan] = []
    seen_pair: set[tuple[str, str]] = set()

    for sent_idx, sent in [(i, s.strip()) for i, s in sentences if s and s.strip()]:
        in_anchor = _sentence_has_anchor(sent, anchors)
        ctx = "anchor" if in_anchor else "routed"
        doc = nlp(sent[:1_000_000])

        root_spans: list[tuple[int, int]] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            et = ent.text.strip()
            if any(a in et or et in a for a in anchors if len(a) >= 2):
                root_spans.append((ent.start_char, ent.end_char))

        for ent in doc.ents:
            mapped = _map_to_seven_labels(ent.label_)
            if mapped is None:
                continue
            # 降噪：非 anchor 句禁用 PERSON（与现有窗口策略一致）
            if not in_anchor and ent.label_ == "PERSON":
                continue
            span = (ent.start_char, ent.end_char)
            if any(_overlap(span, rs) for rs in root_spans) and ent.label_ == "PERSON":
                continue
            mention = _normalize_mention(ent.text)
            if len(mention) < 2:
                continue
            if mapped in _NUMERIC_LABELS:
                key = ("", f"{mapped}:{mention}".lower())
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append(
                    LinkedSpan(
                        sentence_idx=sent_idx,
                        object_qid="",
                        mention=mention,
                        snippet=f"[{source_label}] {sent[:280]}",
                        score=1.0,
                        ner_label=mapped,
                        context=ctx,
                    )
                )
            else:
                qid, sc = link_mention_to_qid(
                    mention,
                    sent,
                    languages=langs,
                    min_score=min_link_score,
                    entity_map_override=entity_map,
                )
                if not qid or qid == ROOT_ENTITY_QID:
                    continue
                key = (qid, mention.lower())
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                out.append(
                    LinkedSpan(
                        sentence_idx=sent_idx,
                        object_qid=qid,
                        mention=mention,
                        snippet=f"[{source_label}] {sent[:280]}",
                        score=sc,
                        ner_label=mapped,
                        context=ctx,
                    )
                )

        for men, lab in _extract_numeric_mentions(sent):
            key = ("", f"{lab}:{men}".lower())
            if key in seen_pair:
                continue
            seen_pair.add(key)
            out.append(
                LinkedSpan(
                    sentence_idx=sent_idx,
                    object_qid="",
                    mention=men,
                    snippet=f"[{source_label}] {sent[:280]}",
                    score=1.0,
                    ner_label=lab,
                    context=ctx,
                )
            )
    return out


def ingest_linked_spans(
    g: GraphBuild,
    spans: list[LinkedSpan],
    *,
    citation_key: str,
    source_url: str,
    root_qid: str = ROOT_ENTITY_QID,
) -> None:
    pred = "cooccurrence_linked"
    ner_kind = {"PER": "Person", "ORG": "Organization", "LOC": "Location"}
    for sp in spans:
        if not sp.object_qid:
            # 日期/货币等字面值不入 Wikidata 节点，仅在 triples 导出中保留
            continue
        g.ensure_node(root_qid)
        # 不用 mention 覆盖节点 name（name 由结构化层 Wikidata label 提供）；mention 仅作为证据属性保留。
        kind = ner_kind.get(str(sp.ner_label).upper().strip(), "")
        g.ensure_node(sp.object_qid, labels=(kind,) if kind else ())
        g.add_edge(
            root_qid,
            sp.object_qid,
            f"EL_{sp.ner_label}",
            f"EL_{sp.ner_label}",
            "OUT",
            provenance="ner_entity_linking",
            citation_key=citation_key,
            snippet=f"{sp.snippet} | ner={sp.ner_label} ctx={sp.context} score={sp.score:.2f}",
            source_url=source_url,
            score=float(sp.score),
            ner_label=sp.ner_label,
            mention=sp.mention[:120],
            mention_context=sp.context,
            predicate=pred,
        )
