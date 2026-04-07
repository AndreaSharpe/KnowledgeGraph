"""将 build_knowledge_graph 的结果导入 Neo4j（需本机 Neo4j 与密码）。"""

from __future__ import annotations

import os
import re
from pathlib import Path

from neo4j import GraphDatabase

from .build import build_knowledge_graph, export_all
from .graph_model import GraphBuild


_REL_SAFE = re.compile(r"^(WIKI_P\d+|EXT_[A-Z0-9_]+)$")


def load_via_driver(g: GraphBuild, uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for n in g.nodes.values():
                session.run(
                    """
                    MERGE (x:Entity {id: $id})
                    SET x.name = $name, x.extra = $extra
                    """,
                    id=n["id"],
                    name=n["name"],
                    extra=n.get("extra") or "",
                )
            for e in g.edges:
                rt = e["rel_type"]
                if not _REL_SAFE.match(rt):
                    raise ValueError(f"非法关系类型: {rt}")
                session.run(
                    f"""
                    MATCH (a:Entity {{id: $sid}}), (b:Entity {{id: $eid}})
                    MERGE (a)-[r:{rt}]->(b)
                    SET r.prop_id = $prop_id,
                        r.prop_label = $prop_label,
                        r.direction = $direction,
                        r.provenance = $provenance,
                        r.citation_key = $citation_key,
                        r.snippet = $snippet,
                        r.source_url = $source_url
                    """,
                    sid=e["start_id"],
                    eid=e["end_id"],
                    prop_id=e["prop_id"],
                    prop_label=e["prop_label"],
                    direction=e["direction"],
                    provenance=e.get("provenance", "wikidata"),
                    citation_key=e.get("citation_key", ""),
                    snippet=e.get("snippet", ""),
                    source_url=e.get("source_url", ""),
                )
    finally:
        driver.close()


def main() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        raise SystemExit("请设置环境变量 NEO4J_PASSWORD")

    root = Path(os.environ.get("TURING_KG_ROOT", ".")).resolve()
    data_dir = Path(os.environ.get("TURING_KG_DATA", str(root / "data")))

    g, triple_rows = build_knowledge_graph(root)
    export_all(root, g, triple_rows)
    load_via_driver(g, uri=uri, user=user, password=password)
    print(f"已写入 Neo4j 并导出 CSV 至 {data_dir.resolve()}")


if __name__ == "__main__":
    main()
