"""将 build_knowledge_graph 的结果导入 Neo4j（需本机 Neo4j 与密码）。"""

from __future__ import annotations

import os
import re
import sys
import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

try:
    # 作为包运行（推荐）：python -m turing_kg.neo4j_loader
    from .build import build_knowledge_graph, export_all
    from .graph_model import GraphBuild
except ImportError:  # pragma: no cover
    # 直接运行文件：python .\turing_kg\neo4j_loader.py
    # 将仓库根目录加入 sys.path 以支持绝对导入
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from turing_kg.build import build_knowledge_graph, export_all  # type: ignore
    from turing_kg.graph_model import GraphBuild  # type: ignore


_REL_SAFE = re.compile(r"^(WIKI_P\d+|EXT_[A-Z0-9_]+)$")
_LABEL_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _try_run(session: Any, cypher: str) -> bool:
    try:
        session.run(cypher)
        return True
    except Exception:
        return False


def _create_constraints_and_indexes(session: Any) -> None:
    # 兼容 Neo4j 4.x/5.x：先尝试新语法，失败再回退旧语法；两者都失败则跳过（不影响导入正确性）。
    ok = _try_run(session, "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    if not ok:
        _try_run(session, "CREATE CONSTRAINT entity_id IF NOT EXISTS ON (n:Entity) ASSERT n.id IS UNIQUE")

    ok = _try_run(session, "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
    if not ok:
        _try_run(session, "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")

    # 关系属性索引：
    # - Neo4j 5 支持 FOR ()-[r]-() ON (r.prop)
    # - Neo4j 4 通常要求指定关系类型 FOR ()-[r:TYPE]-() ON (r.prop)
    # 本项目关系类型很多，不强依赖该索引；尽力而为，失败则跳过。
    _try_run(session, "CREATE INDEX rel_provenance IF NOT EXISTS FOR ()-[r]-() ON (r.provenance)")
    # 用于语义聚合后的边去重（edge_id）
    _try_run(session, "CREATE INDEX rel_edge_id IF NOT EXISTS FOR ()-[r]-() ON (r.edge_id)")


def load_via_driver(g: GraphBuild, uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            _create_constraints_and_indexes(session)

            def _neo4j_safe_props(d: dict[str, Any]) -> dict[str, Any]:
                """
                Neo4j property values must be primitive or arrays thereof.
                Dict/list are serialized to JSON strings to keep information.
                """
                out: dict[str, Any] = {}
                for k, v in (d or {}).items():
                    if v is None:
                        continue
                    if isinstance(v, (dict, list)):
                        out[k] = json.dumps(v, ensure_ascii=False)[:4000]
                    else:
                        out[k] = v
                return out

            for n in g.nodes.values():
                labels = list(n.get("labels") or [])
                # 不依赖 APOC：先 MERGE :Entity，再逐个 SET 标签
                session.run(
                    """
                    MERGE (x:Entity {id: $id})
                    SET x.name = $name, x.extra = $extra
                    """,
                    id=n["id"],
                    name=n["name"],
                    extra=n.get("extra") or "",
                )
                for lab in labels:
                    if lab in ("Entity",) or not _LABEL_SAFE.match(str(lab)):
                        continue
                    session.run(
                        f"""
                        MATCH (x:Entity {{id: $id}})
                        SET x:{lab}
                        """,
                        id=n["id"],
                    )
                # 其它节点属性（扁平化写入）
                extra_props = {k: v for k, v in n.items() if k not in ("id", "name", "extra", "labels")}
                if extra_props:
                    session.run(
                        """
                        MATCH (x:Entity {id: $id})
                        SET x += $props
                        """,
                        id=n["id"],
                        props=_neo4j_safe_props(extra_props),
                    )
            for e in g.edges:
                rt = e["rel_type"]
                if not _REL_SAFE.match(rt):
                    raise ValueError(f"非法关系类型: {rt}")
                base_props = {
                    "prop_id": e.get("prop_id", ""),
                    "prop_label": e.get("prop_label", ""),
                    "direction": e.get("direction", "OUT"),
                    "provenance": e.get("provenance", "wikidata"),
                    "layer": e.get("layer", "facts" if e.get("provenance") == "wikidata" else "evidence"),
                    "citation_key": e.get("citation_key", ""),
                    "snippet": e.get("snippet", ""),
                    "source_url": e.get("source_url", ""),
                }
                extra_props = {
                    k: v
                    for k, v in e.items()
                    if k
                    not in (
                        "start_id",
                        "end_id",
                        "rel_type",
                        "prop_id",
                        "prop_label",
                        "direction",
                        "provenance",
                        "layer",
                        "citation_key",
                        "snippet",
                        "source_url",
                    )
                }
                base_props.update(_neo4j_safe_props(extra_props))
                session.run(
                    f"""
                    MATCH (a:Entity {{id: $sid}}), (b:Entity {{id: $eid}})
                    MERGE (a)-[r:{rt} {{edge_id: $edge_id}}]->(b)
                    SET r += $props
                    """,
                    sid=e["start_id"],
                    eid=e["end_id"],
                    edge_id=str(base_props.get("edge_id") or ""),
                    props=base_props,
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
    mode = os.environ.get("TURING_KG_MODE", "").strip().lower() or None

    # 允许用 TURING_KG_MODE=from_curated/export_only 避免 full 过慢
    g, triple_rows = build_knowledge_graph(root, mode=mode)
    export_all(root, g, triple_rows)
    load_via_driver(g, uri=uri, user=user, password=password)
    print(f"已写入 Neo4j 并导出 CSV 至 {data_dir.resolve()}")


if __name__ == "__main__":
    main()
