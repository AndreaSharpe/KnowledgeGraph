#构建图灵知识图谱并导出到 data。

from __future__ import annotations

from pathlib import Path

from turing_kg.build import build_knowledge_graph, export_all


def main() -> None:
    root = Path(__file__).resolve().parent
    g, triple_rows = build_knowledge_graph(root)
    export_all(root, g, triple_rows)
    print(f"完成。节点 {len(g.nodes)}，边 {len(g.edges)}。输出：{root / 'data'}")


if __name__ == "__main__":
    main()
