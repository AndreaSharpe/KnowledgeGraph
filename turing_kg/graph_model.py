"""属性图模型：节点与边（与 Neo4j 导出列一致）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def rel_type_wikidata_prop(prop_id: str) -> str:
    clean = prop_id.strip().upper().replace(":", "_")
    return f"WIKI_{clean}"


def rel_type_text_predicate(pred_label: str) -> str:
    s = re.sub(r"[^\w]+", "_", pred_label, flags=re.UNICODE).strip("_").upper()
    if not s:
        s = "REL"
    return "EXT_" + s[:36]


@dataclass
class GraphBuild:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def ensure_node(self, qid: str, name: str | None = None, extra: str | None = None) -> None:
        if qid not in self.nodes:
            self.nodes[qid] = {"id": qid, "name": name or qid, "extra": extra or ""}
        else:
            if name and (self.nodes[qid]["name"] == qid or len(name) > len(self.nodes[qid]["name"] or "")):
                self.nodes[qid]["name"] = name
            if extra:
                self.nodes[qid]["extra"] = extra

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
    ) -> None:
        rtype = rel_type_wikidata_prop(prop_id) if provenance == "wikidata" else rel_type_text_predicate(prop_label)
        self.edges.append(
            {
                "start_id": src,
                "end_id": dst,
                "prop_id": prop_id,
                "prop_label": prop_label,
                "rel_type": rtype,
                "direction": direction,
                "provenance": provenance,
                "citation_key": citation_key,
                "snippet": snippet,
                "source_url": source_url,
            }
        )
