"""事件抽取：基于规则的 Trigger → Arguments（事件类型识别 + 论元填充）。

本模块在“已路由到某个 seed 的句子集合”上运行，对每个句子：
- 先命中触发词（trigger）；
- 再从已链接实体（`linked_by_sentence`）或句内槽位中取论元（arguments）；
- 产出 `EventRecord`。

I/O（落盘位置由调用方决定）：
- 输入：`seed_items: list[(sentence_idx, sentence)]`，可选 `linked_by_sentence: {idx: [(qid, mention, ner_label)]}`
- 输出：`list[EventRecord]`；或通过 `ingest_events()` 写入图（Event 节点 + EVENT_ARG 边）

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable

from ..graph_model import GraphBuild
from ..linking.entity_linking import link_mention_to_qid


@dataclass(frozen=True)
class Arg:
    """事件论元。`qid` 为空表示字面值（例如年份）。"""
    role: str
    qid: str  # Q… or "" for literal
    mention: str
    ner_label: str = ""


@dataclass(frozen=True)
class EventRecord:
    """单句级事件记录（证据级）。后续可在 `ingest_events()` 中聚合为 canonical 事件。"""
    event_id: str
    event_type: str
    seed_id: str
    seed_qid: str
    source_id: str
    source_url: str
    source_label: str
    citation_key: str
    sentence_idx: int
    sentence: str
    trigger: str
    trigger_span: tuple[int, int] | None
    args: tuple[Arg, ...]
    method: str = "rule_trigger_args_v1"

    def to_json(self) -> dict[str, Any]:
        """序列化为可落盘 JSON（JSONL 每行一条）。"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "seed_id": self.seed_id,
            "seed_qid": self.seed_qid,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_label": self.source_label,
            "citation_key": self.citation_key,
            "sentence_idx": int(self.sentence_idx),
            "sentence": self.sentence,
            "trigger": self.trigger,
            "trigger_span": list(self.trigger_span) if self.trigger_span else None,
            "args": [
                {"role": a.role, "qid": a.qid, "mention": a.mention, "ner_label": a.ner_label}
                for a in self.args
            ],
            "method": self.method,
        }


_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

# Award 列表句模式：Name (in YEAR)
_RE_AWARDEE_PAREN_YEAR = re.compile(
    r"(?P<name>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)+)\s*\(\s*in\s*(?P<year>(?:19|20)\d{2})\s*\)",
    flags=re.IGNORECASE,
)


def _stable_event_id(parts: Iterable[str]) -> str:
    """用若干字段生成稳定短 ID（用于证据级 event_id）。"""
    basis = "|".join([p.strip() for p in parts if p is not None])
    return "EVT_" + sha1(basis.encode("utf-8")).hexdigest()[:12]


def _is_probably_en(s: str) -> bool:
    """极轻量语言判别：中文字符比例很低则视为英文。"""
    if not s:
        return True
    zh = len(re.findall(r"[\u4e00-\u9fff]", s))
    return (zh / max(len(s), 1)) < 0.08


def _trigger_hit(sentence: str, triggers: tuple[str, ...]) -> tuple[str, tuple[int, int]] | None:
    """在句中命中任意 trigger，返回 (trigger, span)；未命中返回 None。"""
    low = sentence.lower()
    for t in triggers:
        tt = (t or "").strip().lower()
        if not tt:
            continue
        idx = low.find(tt)
        if idx >= 0:
            return t, (idx, idx + len(tt))
    return None


def _dedup_args(args: list[Arg]) -> tuple[Arg, ...]:
    """按 (role, qid, mention) 去重并保持顺序。"""
    seen: set[tuple[str, str, str]] = set()
    out: list[Arg] = []
    for a in args:
        key = (a.role, a.qid, a.mention.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return tuple(out)


def _link_person_name(name: str, sentence: str, *, entity_map: dict[str, dict], min_link_score: float) -> tuple[str, float]:
    """将英文人名链接到 Wikidata QID（必要时用 entity_map 作为优先覆盖）。"""
    return link_mention_to_qid(
        name,
        sentence,
        languages=("en", "zh") if _is_probably_en(sentence) else ("zh", "en"),
        min_score=min_link_score,
        entity_map_override=entity_map,
    )


def extract_award_events_from_sentence(
    sentence: str,
    *,
    sentence_idx: int,
    seed_id: str,
    seed_qid: str,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    entity_map: dict[str, dict],
    linked_entities: list[tuple[str, str, str]],  # (qid, mention, ner_label)
    min_link_score: float = 0.14,
) -> list[EventRecord]:
    """抽取 AwardEvent：优先匹配“Name (in YEAR)”列表句，其次从已链接 PER + 年份回退。"""
    triggers_en = ("have been awarded", "was awarded", "were awarded", "won", "recipient", "awarded", "honor", "prize")
    triggers_zh = ("获奖", "得主", "授予", "颁发")
    triggers = triggers_en if _is_probably_en(sentence) else triggers_zh
    th = _trigger_hit(sentence, triggers)
    if not th:
        return []
    trig, trig_span = th

    # 1) 强模式：Name (in YEAR)
    pairs = [(m.group("name").strip(), m.group("year").strip()) for m in _RE_AWARDEE_PAREN_YEAR.finditer(sentence)]
    out: list[EventRecord] = []
    if pairs:
        for name, year in pairs:
            qid, sc = _link_person_name(name, sentence, entity_map=entity_map, min_link_score=min_link_score)
            if not qid or not qid.startswith("Q"):
                continue
            args: list[Arg] = [
                Arg(role="Recipient", qid=qid, mention=name, ner_label="PER"),
                Arg(role="Award", qid=seed_qid, mention=seed_qid, ner_label=""),
                Arg(role="Time", qid="", mention=year, ner_label="DATE"),
            ]
            eid = _stable_event_id(
                [
                    seed_id,
                    source_id,
                    str(sentence_idx),
                    "AwardEvent",
                    trig,
                    qid,
                    year,
                    seed_qid,
                ]
            )
            out.append(
                EventRecord(
                    event_id=eid,
                    event_type="AwardEvent",
                    seed_id=seed_id,
                    seed_qid=seed_qid,
                    source_id=source_id,
                    source_url=source_url,
                    source_label=source_label,
                    citation_key=citation_key,
                    sentence_idx=sentence_idx,
                    sentence=sentence,
                    trigger=trig,
                    trigger_span=trig_span,
                    args=_dedup_args(args),
                    method="award_list_paren_year",
                )
            )
        if out:
            return out

    # 2) 回退：从链接实体里抓 PER + 年份
    years = [y.group(1) for y in _RE_YEAR.finditer(sentence)]
    per_qids = [(qid, men) for (qid, men, lab) in linked_entities if qid.startswith("Q") and lab == "PER"]
    if not per_qids:
        return []
    year = years[0] if years else ""
    for qid, men in per_qids[:2]:
        args2: list[Arg] = [
            Arg(role="Recipient", qid=qid, mention=men, ner_label="PER"),
            Arg(role="Award", qid=seed_qid, mention=seed_qid, ner_label=""),
        ]
        if year:
            args2.append(Arg(role="Time", qid="", mention=year, ner_label="DATE"))
        eid = _stable_event_id([seed_id, source_id, str(sentence_idx), "AwardEvent", trig, qid, year, seed_qid])
        out.append(
            EventRecord(
                event_id=eid,
                event_type="AwardEvent",
                seed_id=seed_id,
                seed_qid=seed_qid,
                source_id=source_id,
                source_url=source_url,
                source_label=source_label,
                citation_key=citation_key,
                sentence_idx=sentence_idx,
                sentence=sentence,
                trigger=trig,
                trigger_span=trig_span,
                args=_dedup_args(args2),
                method="award_linked_per_year_fallback",
            )
        )
    return out


def extract_employment_or_education_events_from_sentence(
    sentence: str,
    *,
    sentence_idx: int,
    seed_id: str,
    seed_qid: str,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    linked_entities: list[tuple[str, str, str]],  # (qid, mention, ner_label)
) -> list[EventRecord]:
    """抽取 Employment/Education：触发词命中后从已链接实体中填充 Person/Organization/Time。"""
    s = sentence.strip()
    if not s:
        return []

    is_en = _is_probably_en(s)
    trig_emp = ("worked at", "joined", "employed by", "professor at", "lecturer at", "served at")
    trig_edu = ("studied at", "educated at", "graduated from", "attended", "student at")
    trig_emp_zh = ("任职", "就职", "受雇", "加入", "在", "工作于")
    trig_edu_zh = ("就读", "毕业", "入学", "在", "学习于")

    event_type = ""
    th = _trigger_hit(s, trig_emp if is_en else trig_emp_zh)
    if th:
        event_type = "EmploymentEvent"
    else:
        th = _trigger_hit(s, trig_edu if is_en else trig_edu_zh)
        if th:
            event_type = "EducationEvent"
    if not th or not event_type:
        return []
    trig, trig_span = th

    years = [y.group(1) for y in _RE_YEAR.finditer(s)]
    year = years[0] if years else ""

    # org: pick first linked ORG
    orgs = [(qid, men) for (qid, men, lab) in linked_entities if qid.startswith("Q") and lab == "ORG"]
    if not orgs:
        return []
    org_qid, org_men = orgs[0]

    # person: prefer seed if it's a person seed, else first linked PER
    person_qid = seed_qid if (seed_id == "turing_person" and seed_qid.startswith("Q")) else ""
    person_men = ""
    if not person_qid:
        pers = [(qid, men) for (qid, men, lab) in linked_entities if qid.startswith("Q") and lab == "PER"]
        if pers:
            person_qid, person_men = pers[0]
    if not person_qid:
        return []

    args: list[Arg] = [
        Arg(role="Person", qid=person_qid, mention=person_men or person_qid, ner_label="PER"),
        Arg(role="Organization", qid=org_qid, mention=org_men, ner_label="ORG"),
    ]
    if year:
        args.append(Arg(role="Time", qid="", mention=year, ner_label="DATE"))

    eid = _stable_event_id([seed_id, source_id, str(sentence_idx), event_type, trig, person_qid, org_qid, year])
    return [
        EventRecord(
            event_id=eid,
            event_type=event_type,
            seed_id=seed_id,
            seed_qid=seed_qid,
            source_id=source_id,
            source_url=source_url,
            source_label=source_label,
            citation_key=citation_key,
            sentence_idx=sentence_idx,
            sentence=s,
            trigger=trig,
            trigger_span=trig_span,
            args=_dedup_args(args),
            method="employment_education_rule_v1",
        )
    ]


def extract_publication_or_proposal_events_from_sentence(
    sentence: str,
    *,
    sentence_idx: int,
    seed_id: str,
    seed_qid: str,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    entity_map: dict[str, dict],
    linked_entities: list[tuple[str, str, str]],
    min_link_score: float,
) -> list[EventRecord]:
    """抽取 PublicationEvent：触发词命中后填充 Person + WorkOrConcept（链接优先，否则字面值）+ Time。"""
    s = sentence.strip()
    if not s:
        return []
    is_en = _is_probably_en(s)
    triggers = ("proposed", "introduced", "published", "paper", "article")
    triggers_zh = ("提出", "发表", "发表于", "论文", "文章")
    th = _trigger_hit(s, triggers if is_en else triggers_zh)
    if not th:
        return []
    trig, trig_span = th

    years = [y.group(1) for y in _RE_YEAR.finditer(s)]
    year = years[0] if years else ""

    # person
    person_qid = seed_qid if (seed_id == "turing_person" and seed_qid.startswith("Q")) else ""
    person_men = ""
    if not person_qid:
        pers = [(qid, men) for (qid, men, lab) in linked_entities if qid.startswith("Q") and lab == "PER"]
        if pers:
            person_qid, person_men = pers[0]
    if not person_qid:
        return []

    # work/concept
    obj_qid = ""
    obj_men = ""
    # 若 seed 自身是概念（如 turing_machine），允许将 seed_qid 作为 WorkOrConcept。
    if seed_id == "turing_machine" and seed_qid.startswith("Q"):
        obj_qid = seed_qid
        obj_men = seed_qid
    else:
        # 优先：从触发词后的槽位抓取短语并尝试链接（英文/中文）。
        m = None
        if is_en:
            m = re.search(
                r"\b(?:proposed|introduced|published)\b\s+(?:the\s+|a\s+|an\s+)?(?P<obj>[^,.;]{3,80})",
                s,
                flags=re.IGNORECASE,
            )
        else:
            m = re.search(r"(?:提出|发表|发表于)(?P<obj>[^。；，]{2,60})", s)
        if m:
            obj_men = (m.group("obj") or "").strip().strip("\"'“”‘’()（）[]【】")
            if obj_men:
                qid, sc = link_mention_to_qid(
                    obj_men,
                    s,
                    languages=("en", "zh") if is_en else ("zh", "en"),
                    min_score=min_link_score,
                    entity_map_override=entity_map,
                )
                if qid and qid.startswith("Q"):
                    obj_qid = qid
        # 回退：从已链接实体里挑一个非 person 的 ORG/LOC 充当对象。
        if not obj_qid:
            for qid, men, lab in linked_entities:
                if not qid.startswith("Q"):
                    continue
                if qid == person_qid:
                    continue
                if lab in ("ORG", "LOC"):
                    obj_qid = qid
                    obj_men = men
                    break

    args: list[Arg] = [Arg(role="Person", qid=person_qid, mention=person_men or person_qid, ner_label="PER")]
    if obj_qid:
        args.append(Arg(role="WorkOrConcept", qid=obj_qid, mention=obj_men or obj_qid, ner_label=""))
    elif obj_men:
        args.append(Arg(role="WorkOrConcept", qid="", mention=obj_men, ner_label=""))
    else:
        return []
    if year:
        args.append(Arg(role="Time", qid="", mention=year, ner_label="DATE"))

    eid = _stable_event_id([seed_id, source_id, str(sentence_idx), "PublicationEvent", trig, person_qid, obj_qid or obj_men, year])
    return [
        EventRecord(
            event_id=eid,
            event_type="PublicationEvent",
            seed_id=seed_id,
            seed_qid=seed_qid,
            source_id=source_id,
            source_url=source_url,
            source_label=source_label,
            citation_key=citation_key,
            sentence_idx=sentence_idx,
            sentence=s,
            trigger=trig,
            trigger_span=trig_span,
            args=_dedup_args(args),
            method="publication_proposal_rule_v1",
        )
    ]


def extract_events_from_sentences(
    seed_items: list[tuple[int, str]],
    *,
    seed_id: str,
    seed_qid: str,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    entity_map: dict[str, dict],
    # map: sentence_idx -> list[(qid, mention, ner_label)]
    linked_by_sentence: dict[int, list[tuple[str, str, str]]] | None = None,
    min_link_score: float = 0.14,
) -> list[EventRecord]:
    """对 routed 句子集合做事件抽取，返回证据级 `EventRecord` 列表。"""
    out: list[EventRecord] = []
    linked_by_sentence = linked_by_sentence or {}
    for si, sent in seed_items:
        s = (sent or "").strip()
        if not s:
            continue
        linked = linked_by_sentence.get(int(si), [])
        if seed_id in ("turing_award", "turing_person", "turing_machine"):
            out.extend(
                extract_award_events_from_sentence(
                    s,
                    sentence_idx=int(si),
                    seed_id=seed_id,
                    seed_qid=seed_qid,
                    source_id=source_id,
                    source_url=source_url,
                    source_label=source_label,
                    citation_key=citation_key,
                    entity_map=entity_map,
                    linked_entities=linked,
                    min_link_score=min_link_score,
                )
            )
        out.extend(
            extract_employment_or_education_events_from_sentence(
                s,
                sentence_idx=int(si),
                seed_id=seed_id,
                seed_qid=seed_qid,
                source_id=source_id,
                source_url=source_url,
                source_label=source_label,
                citation_key=citation_key,
                linked_entities=linked,
            )
        )
        out.extend(
            extract_publication_or_proposal_events_from_sentence(
                s,
                sentence_idx=int(si),
                seed_id=seed_id,
                seed_qid=seed_qid,
                source_id=source_id,
                source_url=source_url,
                source_label=source_label,
                citation_key=citation_key,
                entity_map=entity_map,
                linked_entities=linked,
                min_link_score=min_link_score,
            )
        )
    return out


def ingest_events(g: GraphBuild, events: list[EventRecord]) -> None:
    """
    将事件写入 GraphBuild：Event 节点 + 论元边（EVENT_ARG）。

    处理步骤：
    - 证据级事件 -> canonical 合并：按 (event_type + 关键论元 + time) 聚合为“同一事实”；
    - canonical Event 节点：`EVT_CAN_*`，`extra` 中保存（截断后的）证据列表；
    - 论元边：Event -> QID 或 Event -> Literal（LIT_*）。
    """
    def _norm_literal(s: str) -> str:
        """规范化字面值，用于 canonical key 与 Literal 节点 ID。"""
        t = (s or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = t.strip("\"'“”‘’()（）[]【】《》")
        return t

    def _arg_value_for_key(a: Arg) -> str:
        """canonical key 使用：QID 直接用 QID，字面值用规范化 mention。"""
        if a.qid and a.qid.startswith("Q"):
            return a.qid
        return _norm_literal(a.mention)

    def _first_time(ev: EventRecord) -> str:
        """取事件中的第一个 Time 论元（若有）。"""
        for a in ev.args:
            if (a.role or "").strip().lower() == "time":
                v = _norm_literal(a.mention)
                if v:
                    return v
        return ""

    def _canonical_key(ev: EventRecord) -> tuple[str, str, str, str, str]:
        """将证据级事件映射为 canonical 合并 key（同一事实应得到同一 key）。"""
        et = (ev.event_type or "").strip() or "Event"
        def pick(role: str) -> str:
            for a in ev.args:
                if (a.role or "").strip().lower() == role.lower():
                    v = _arg_value_for_key(a)
                    if v:
                        return v
            return ""

        if et == "AwardEvent":
            return (et, pick("Recipient"), pick("Award"), _first_time(ev), "")
        if et in ("EmploymentEvent", "EducationEvent"):
            return (et, pick("Person"), pick("Organization"), _first_time(ev), "")
        if et == "PublicationEvent":
            return (et, pick("Person"), pick("WorkOrConcept"), _first_time(ev), "")
        # 回退：取前两个论元值作为 key（尽量合并，但不追求覆盖所有类型）。
        av = [_arg_value_for_key(a) for a in ev.args if _arg_value_for_key(a)]
        a1 = av[0] if len(av) > 0 else ""
        a2 = av[1] if len(av) > 1 else ""
        return (et, a1, a2, _first_time(ev), "")

    # 分组：证据级事件 -> canonical 事件
    by_key: dict[tuple[str, str, str, str, str], list[EventRecord]] = {}
    for ev in events:
        if not ev or not ev.event_type:
            continue
        k = _canonical_key(ev)
        if not k[0] or not k[1] or not k[2]:
            # require at least event_type + two key args
            continue
        by_key.setdefault(k, []).append(ev)

    for k, evs in by_key.items():
        et, a1, a2, tval, _x = k
        can_id = "EVT_CAN_" + sha1("|".join([et, a1, a2, tval]).encode("utf-8")).hexdigest()[:12]
        # 合并论元（同 role + value 去重）
        args_seen: set[tuple[str, str, str]] = set()
        args_union: list[Arg] = []
        for ev in evs:
            for a in ev.args:
                role = (a.role or "Arg").strip()
                v = _arg_value_for_key(a)
                key2 = (role, str(a.qid or ""), v)
                if key2 in args_seen:
                    continue
                args_seen.add(key2)
                args_union.append(a)

        # 证据列表（上限 10 条，避免节点属性过大）
        evid: list[dict[str, Any]] = []
        for ev in evs[:10]:
            evid.append(
                {
                    "source_id": ev.source_id,
                    "sentence_idx": int(ev.sentence_idx),
                    "trigger": ev.trigger,
                    "snippet": (ev.sentence or "")[:280],
                    "citation_key": ev.citation_key,
                    "source_url": ev.source_url,
                    "seed_id": ev.seed_id,
                    "seed_qid": ev.seed_qid,
                    "method": ev.method,
                }
            )

        ev_name = f"{et}:{(evs[0].trigger or '').strip()}".strip(":")[:120]
        extra = json.dumps({"canonical_key": k, "evidence": evid}, ensure_ascii=False)[:1800]
        g.ensure_node(can_id, name=ev_name, extra=extra, labels=("Event", et))
        g.ensure_node(can_id, props={"event_type": et, "evidence_count": len(evs)})

        # 写入论元边：QID 直接连；字面值建 Literal 节点再连
        for a in _dedup_args(args_union):
            role = (a.role or "Arg").strip() or "Arg"
            if a.qid and a.qid.startswith("Q"):
                g.ensure_node(a.qid)
                if g.has_edge(can_id, a.qid, "EVENT_ARG", provenance="event_extraction"):
                    continue
                g.add_edge(
                    can_id,
                    a.qid,
                    prop_id="EVENT_ARG",
                    prop_label="EVENT_ARG",
                    direction="OUT",
                    provenance="event_extraction",
                    citation_key=(evs[0].citation_key or ""),
                    snippet=(evs[0].sentence or "")[:280],
                    source_url=(evs[0].source_url or ""),
                    role=role,
                    arg_mention=(a.mention or "")[:120],
                    arg_ner_label=a.ner_label or "",
                    event_type=et,
                )
            else:
                lit = (a.mention or "").strip()
                if not lit:
                    continue
                lid = "LIT_" + sha1(f"{role}|{_norm_literal(lit)}".encode("utf-8")).hexdigest()[:12]
                g.ensure_node(lid, name=lit[:120], labels=("Literal",))
                if g.has_edge(can_id, lid, "EVENT_ARG", provenance="event_extraction"):
                    continue
                g.add_edge(
                    can_id,
                    lid,
                    prop_id="EVENT_ARG",
                    prop_label="EVENT_ARG",
                    direction="OUT",
                    provenance="event_extraction",
                    citation_key=(evs[0].citation_key or ""),
                    snippet=(evs[0].sentence or "")[:280],
                    source_url=(evs[0].source_url or ""),
                    role=role,
                    arg_mention=lit[:200],
                    arg_ner_label=a.ner_label or "",
                    event_type=et,
                )

