"""属性图模型：节点与边（与 Neo4j 导出列一致）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Iterable


def rel_type_wikidata_prop(prop_id: str) -> str:
    clean = prop_id.strip().upper().replace(":", "_")
    return f"WIKI_{clean}"


def rel_type_text_predicate(pred_label: str) -> str:
    s = re.sub(r"[^\w]+", "_", pred_label, flags=re.UNICODE).strip("_").upper()
    if not s:
        s = "REL"
    return "EXT_" + s[:36]


_WD_PROP_RE = re.compile(r"^P\d+$", flags=re.IGNORECASE)


@dataclass
class GraphBuild:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def ensure_node(
        self,
        qid: str,
        name: str | None = None,
        extra: str | None = None,
        *,
        labels: Iterable[str] = (),
        props: dict[str, Any] | None = None,
    ) -> None:
        if qid not in self.nodes:
            self.nodes[qid] = {"id": qid, "name": name or qid, "extra": extra or "", "labels": sorted({x for x in labels if x})}
            if props:
                for k, v in props.items():
                    if k in ("id", "name"):
                        continue
                    self.nodes[qid][k] = v
        else:
            # name 只在当前还不可读（等于 qid/为空）时才允许更新。
            # 不允许用更长的 mention 覆盖更干净的 Wikidata label，避免节点名变成短句/片段。
            cur_name = str(self.nodes[qid].get("name") or "")
            if name and (not cur_name or cur_name == qid):
                self.nodes[qid]["name"] = name
            if extra:
                self.nodes[qid]["extra"] = extra
            if labels:
                cur = set(self.nodes[qid].get("labels") or [])
                cur.update([x for x in labels if x])
                self.nodes[qid]["labels"] = sorted(cur)
            if props:
                for k, v in props.items():
                    if k in ("id", "name") or v is None:
                        continue
                    self.nodes[qid][k] = v

    def has_edge(self, src: str, dst: str, prop_id: str, *, provenance: str | None = None) -> bool:
        """用于去重/分层策略：可选按 provenance 精确匹配。"""
        for e in self.edges:
            if e.get("start_id") != src or e.get("end_id") != dst or e.get("prop_id") != prop_id:
                continue
            if provenance is not None and e.get("provenance") != provenance:
                continue
            return True
        return False

    def add_edge(
        self,
        src: str,
        dst: str,
        prop_id: str,
        prop_label: str,
        direction: str,
        *,
        provenance: str = "wikidata",
        citation_key: str = "",
        snippet: str = "",
        source_url: str = "",
        **props: Any,
    ) -> None:
        # 经典 KG：关系类型优先对齐 Wikidata 属性（便于统一查询），来源用 provenance 区分。
        pid = (prop_id or "").strip()
        if _WD_PROP_RE.match(pid):
            rtype = rel_type_wikidata_prop(pid)
        else:
            rtype = rel_type_text_predicate(prop_label)

        # 生成稳定 edge_id（用于 Neo4j MERGE 去重）
        # - facts：同 (src,dst,prop_id) 固定
        # - evidence：同 (src,dst,prop_id,provenance,citation/source/snippet/...) 固定；不同证据保留多条
        if "edge_id" not in props or not props.get("edge_id"):
            basis = "|".join(
                [
                    str(provenance),
                    str(src),
                    str(dst),
                    str(prop_id),
                    str(citation_key),
                    str(source_url),
                    str(snippet)[:400],
                    str(props.get("bag_id", "")),
                    str(props.get("seed_id", "")),
                    str(props.get("seed_type", "")),
                    str(props.get("model", "")),
                ]
            )
            props["edge_id"] = sha1(basis.encode("utf-8")).hexdigest()[:12]

        rec: dict[str, Any] = {
            "start_id": src,
            "end_id": dst,
            "prop_id": prop_id,
            "prop_label": prop_label,
            "rel_type": rtype,
            "direction": direction,
            "provenance": provenance,
            "layer": "facts"
            if provenance in ("wikidata", "wikidata_incoming")
            else "evidence",
            "citation_key": citation_key,
            "snippet": snippet,
            "source_url": source_url,
        }
        for k, v in props.items():
            if v is None:
                continue
            rec[k] = v
        self.edges.append(rec)
