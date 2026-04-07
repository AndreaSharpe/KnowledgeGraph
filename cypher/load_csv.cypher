// 1) 将 data/nodes.csv 与 data/relationships.csv 复制到 Neo4j 的 import 目录。
// 2) 推荐直接用 Python 导入（支持动态关系类型，无需 APOC）：
//      set NEO4J_PASSWORD=...
//      python -m turing_kg.neo4j_loader
//
// 若坚持用纯 Cypher + LOAD CSV，需要为每种 :TYPE 写一条语句，或安装 APOC 后用 apoc.create.relationship。
// 节点示例：
//
// LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
// MERGE (n:Entity {id: row.`id:ID`})
// SET n.name = row.name, n.extra = coalesce(row.extra, '');
