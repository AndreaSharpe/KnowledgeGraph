"""仅刷新图灵机(Q163310)/图灵奖(Q185667)的 Wikidata 入边，不跑 NER/全文构建。

在已有 data/nodes.csv 与 data/relationships.csv 上：
1. 删除旧版 provenance=wikidata_incoming 的边（可反复执行、避免重复）；
2. 调用 Wikidata Query Service 拉入边并写回图；
3. 覆写 data/*.csv（三元组表不变）。

用法（在项目根目录）：

  .\\.venv\\Scripts\\python.exe scripts\\refresh_wikidata_focal_incoming.py

若需强制无视本地 WDQS 缓存、重新打查询，可先删 data/cache/wdqs_*.json
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from turing_kg.io.export_io import write_graph_csv_json
    from turing_kg.io.graph_csv_io import load_graph_build_from_data_csv
    from turing_kg.io.sources_io import load_bibliography
    from turing_kg.structured.wikidata_layer import ingest_focal_incoming_bundles

    data_dir = root / "data"
    g = load_graph_build_from_data_csv(data_dir)
    before = len(g.edges)
    g.edges = [e for e in g.edges if e.get("provenance") != "wikidata_incoming"]
    removed = before - len(g.edges)
    ingest_focal_incoming_bundles(g)
    after = len(g.edges)
    bib_path = root / "sources" / "bibliography.json"
    bib = load_bibliography(bib_path) if bib_path.is_file() else {}
    write_graph_csv_json(g, data_dir, bibliography=bib or None)
    print(
        f"已移除旧入边 {removed} 条，刷新后总边数 {after}，节点 {len(g.nodes)}。"
        f" 输出：{data_dir / 'nodes.csv'}、{data_dir / 'relationships.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
