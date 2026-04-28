// 运行前提：已将 data/nodes.csv + data/relationships.csv 导入 Neo4j（或使用 python -m turing_kg.neo4j_loader）。
// 注意：本项目关系类型默认形如 WIKI_P27 / WIKI_P19 / EXT_*，来源用 r.provenance 区分。

// 1) 找到图灵节点（按 id 或 name）
MATCH (t:Entity)
WHERE t.id = "Q7251" OR toLower(t.name) CONTAINS "turing"
RETURN t
LIMIT 20;

// 2) 查看图灵的 Wikidata 事实边
MATCH (t:Entity {id:"Q7251"})-[r]->(o:Entity)
WHERE r.provenance = "wikidata"
RETURN type(r) AS relType, r.prop_id AS propId, r.prop_label AS propLabel, o.id AS objId, o.name AS objName
ORDER BY relType
LIMIT 100;

// 3) 查看图灵的“文本证据边”（RE / pattern / NER-EL）
MATCH (t:Entity {id:"Q7251"})-[r]->(o:Entity)
WHERE r.provenance <> "wikidata"
RETURN type(r) AS relType, r.provenance AS prov, r.score AS score, r.tau AS tau, r.prop_id AS propId, o.name AS objName, r.snippet AS snippet
ORDER BY coalesce(r.score, 0.0) DESC
LIMIT 100;

// 4) 找出“事实层已存在、同时有证据支持”的边（支持证据）
MATCH (a:Entity)-[r]->(b:Entity)
WHERE r.provenance = "text_re" AND coalesce(r.is_supporting_fact, 0) = 1
RETURN a.name AS subj, type(r) AS relType, r.prop_id AS propId, b.name AS obj, r.score AS score, r.snippet AS snippet
ORDER BY coalesce(r.score, 0.0) DESC
LIMIT 100;

// 5) 按 provenance 统计证据边规模
MATCH ()-[r]->()
RETURN r.provenance AS provenance, count(r) AS n
ORDER BY n DESC;

