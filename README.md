# 图灵知识图谱 — Turing Knowledge Graph（多中心可迁移研究框架）

> **知识工程课程项目**  
> 以艾伦·图灵为核心，构建**多中心种子（Person / Concept / Award）**知识图谱。  
> 强调**全链路可审计中间层**、**Wikidata结构化底座**、**混合抽取策略（规则+DS+PCNN+MIL-Attention）**、**Neo4j友好导出**，支持从已标注中间层快速重放构建。

---

## 一、项目目标与核心特色

本项目不是简单的“爬取-抽取-可视化”演示，而是**一个可迁移、可审计、可训练**的知识图谱构建框架。主要创新点包括：

- **多中心种子设计**：同时支持 `turing_person` (Q7251)、`turing_machine` (Q163310)、`turing_award` (Q185667)，通过 `sources/seed_entities.json` 配置驱动。
- **强审计中间层**：从句子路由、NER mention、候选生成、协同链接（Collective EL）、bag构造、DS标签生成，到MIL预测，全程可追溯。
- **混合知识获取**：Wikidata结构化声明 + Wikipedia多seed摘要 + 本地文章/PDF/Web抓取。
- **混合抽取引擎**：
  - NER：spaCy + CRF（可按source配置切换）
  - 实体链接：候选生成 + Collective Consistency Decoding
  - 关系抽取：规则模式 + 远程监督（DS） + PCNN + MIL-Attention（支持训练/推理分离）
- **导出导向**：生成 `nodes.csv`、`relationships.csv`、`triples_extracted.csv` 及 MIL增强版本，直接支持Neo4j导入。

---

## 二、项目结构（当前真实结构）

```bash
KnowledgeGraph/
├── README.md                       # 本文档（新版）
├── run.py                          # 唯一推荐入口，支持三种构建模式
├── requirements.txt
├── docs/                           # 设计文档与旧版README
│   ├── README.md                   # （对方项目复制版，已被新版替代）
│   └── IMPLEMENTATION_*.md
├── sources/                        # 配置中心（建议后续统一为 config/）
│   ├── seed_entities.json          # 多中心种子定义（anchors、triggers、type）
│   ├── web_sources.json
│   ├── pdf_sources.json
│   ├── bibliography.json
│   └── *config*.json
├── turing_kg/                      # 核心包（当前最核心实现）
│   ├── build.py                    # 核心构建流水线（600+行，建议后续拆分）
│   ├── graph_model.py              # 自定义属性图 GraphBuild
│   ├── config.py
│   ├── structured/                 # Wikidata结构化层
│   ├── sources/                    # 文本采集逻辑
│   ├── extraction/                 # NER、CRF、规则模式、事件抽取
│   ├── linking/                    # 实体链接 + Collective EL
│   ├── attribution/                # 多seed路由、归因、triple合并
│   ├── relation/                   # DS、bag构造、PCNN+MIL-Attention全套
│   ├── io/                         # 中间层读写与导出工具
│   └── neo4j_loader.py
├── scripts/                        # 辅助脚本（训练、重建、评估、抓取）
│   ├── fetch_web_sources.py
│   ├── build_relation_bags.py
│   ├── build_ds_labels.py
│   ├── train_relation_pcnn.py
│   ├── infer_relation_pcnn.py
│   ├── export_mil_triples.py
│   └── rebuild_*.py（多种重放工具）
├── raw/                            # 原始语料
│   └── excerpts/articles/*.txt + book_excerpt.txt + PDF
├── data/                           # 核心输出与中间层
│   ├── curated/                    # 可审计中间层（最重要）
│   │   ├── mentions.jsonl
│   │   ├── resolved.jsonl
│   │   ├── routing.jsonl
│   │   ├── re_predictions_pcnn_*.jsonl
│   │   ├── bags.jsonl
│   │   └── ds_labels.jsonl
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── triples_extracted.csv
│   ├── triples_mil_extracted.csv
│   └── sentence_attribution.jsonl
├── models/                         # CRF等模型
├── cypher/                         # Neo4j导入Cypher脚本
└── .venv/
```

### 项目结构评估

**存在中等程度的结构混乱问题**（诚实说明）：

**问题**：
1. `turing_kg/build.py` 过长（600+行），职责不单一。
2. 配置分散在 `sources/` 多个JSON文件中，缺少统一的 `config/` 目录。
3. `scripts/` 与 `turing_kg/` 部分逻辑有重复。
4. Windows路径导致部分文件同时存在 `\` 和 `/` 版本。
5. 文档碎片化（多个 `IMPLEMENTATION_*.md`）。

**优势**：
- `run.py` 入口设计优秀，支持三种模式（full / from_curated / export_only）。
- 中间层设计非常专业（可审计性远超一般课程项目）。
- MIL训练推理闭环完整，研究价值高。

**改进建议**（可后续快速迭代）：
- 新建 `config/` 目录，统一所有配置。
- 将 `build.py` 拆分为 orchestrator + 各阶段 pipeline。
- 增加 `Makefile` 或统一CLI入口。
- 补充可视化与Web查询层（当前相对薄弱）。

---

## 三、完整构建流程（基于真实代码）

```text
python run.py [--mode full|export_only|from_curated]

1. 模式加载（sources/build_config.json 或命令行参数）
2. Wikidata结构化层（turing_kg/structured/wikidata_layer.py）
   - 多seed声明合并，构建事实底座
3. 文本采集（turing_kg/sources/text_sources.py）
   - Wikipedia多seed摘要 + 文章目录 + book_excerpt + PDF抽页
4. 句子路由与归因（turing_kg/attribution/seed_router.py）
   - 生成 routing.jsonl 与 sentence_attribution.jsonl
5. NER + 实体链接（extraction/ner_link.py + linking/collective_linking.py）
   - 支持spaCy/CRF切换 + Collective EL
6. 关系抽取
   - 规则模式（relation_patterns.py）
   - DS + PCNN + MIL-Attention（relation/ 目录全套：bags、ds_labels、pcnn_mil、infer）
7. 合并与导出（build.py: merge_triple_rows + export_all）
   - MIL结果自动合并到 triples_mil_extracted.csv
   - 输出 nodes.csv / relationships.csv / graph_summary.json
   - 支持 Neo4j 导入
```

### 推荐运行命令

```bash
# 1. 完整构建（首次推荐）
python run.py --mode full

# 2. 使用已有的curated中间层快速重放（调试推荐）
python run.py --mode from_curated

# 3. 仅更新MIL关系并导出（增量实验）
python run.py --mode export_only

# 4. 单独研究MIL流程
python scripts/build_relation_bags.py
python scripts/build_ds_labels.py
python scripts/train_relation_pcnn.py --seed-type Person
python scripts/infer_relation_pcnn.py
python scripts/export_mil_triples.py
```

---

## 四、技术亮点详解（忠实于当前实现）

### 4.1 多中心种子系统
- `sources/seed_entities.json` 统一定义三个种子，包括中英文 `anchors`、`triggers` 和 `type`。
- `attribution/seed_router.py` 实现可解释的句子路由，支持多归因、窗口扩展、结构化 `reasons`。

### 4.2 结构化 + 文本混合建模
- `structured/wikidata_layer.py` 批量获取多seed的claims，构建事实层。
- 文本层通过 `text_sources.py` 灵活汇聚多源语料。

### 4.3 NER与实体链接
- 支持按 `extraction_profile` 配置切换 spaCy/CRF 后端。
- `linking/collective_linking.py` 实现一致性解码，提升跨句实体统一性。
- 完整记录 `mentions`、`candidates`、`resolved` 三层决策过程。

### 4.4 关系抽取（核心研究模块）
- **规则层**：`extraction/relation_patterns.py` + `event_patterns.py`。
- **学习层**（`relation/` 目录）：
  - `ds_dataset.py`、`bags.py`：构造高质量 bag（支持中文主干过滤）。
  - `ds_labels.py`：基于 Wikidata 声明生成远程监督标签（per-seed_type allowlist）。
  - `pcnn_mil.py`：PCNN + MIL-Attention 实现（支持 selective attention）。
  - 训练/推理分离，输出 `re_predictions_pcnn_*.jsonl`。
- `mil_ingest.py` 将高置信预测转换为图边，与规则结果融合去重。

### 4.5 图谱模型与导出
- `graph_model.GraphBuild`：支持丰富节点标签、边 provenance、citation_key、snippet、stable edge_id。
- `io/export_io.py`：生成标准CSV + JSON + summary。
- `neo4j_loader.py` + `cypher/load_csv.cypher` 提供完整导入方案。

---

## 五、环境要求

**Python 版本**：3.10+

**安装命令**：

```bash
conda create -n turing-kg python=3.10 -y
conda activate turing-kg
pip install -r requirements.txt
python -m spacy download zh_core_web_sm en_core_web_sm
```

**关键依赖**：
- `torch`（MIL训练）
- `spacy`（NER）
- `neo4j`（可选导入）
- `scikit-learn`（CRF）

---

## 六、数据产物概览（运行后可在 data/ 查看）

- **中间审计层**（curated/）：routing、mentions、resolved、bags、ds_labels、re_predictions_pcnn_*
- **最终导出**：nodes.csv（366行示例）、relationships.csv、triples_extracted.csv、triples_mil_extracted.csv
- **Neo4j导入**：cypher/load_csv.cypher

---

## 七、开发进度与改进路线

**已完成（当前真实状态）**：
- [x] 多中心种子路由与完整审计链
- [x] Wikidata结构化底座 + 混合文本采集
- [x] 可切换NER后端 + Collective EL
- [x] 规则 + DS + PCNN+MIL-Attention 训练推理闭环
- [x] 丰富导出与Neo4j支持
- [x] from_curated 快速重放模式

**优先改进建议**（按重要性排序）：
1. **重构项目结构**（最高优先）：拆分 `build.py`，统一 `config/` 目录。
2. **补充可视化层**：增加 pyvis/HTML 或 Neo4j Browser 查询模板（当前相对对方项目最明显的短板）。
3. **完善评估体系**：补充 Gold 集 + 自动化阈值校准脚本。
4. **统一CLI入口**：使用 `typer` 或 `click` 提供更友好的命令行体验。
5. **文档与测试**：补充单元测试与更完整的运行示例。

---

## 八、参考资料

- Wikidata：Q7251 (Alan Turing)、Q163310 (Turing Machine)、Q185667 (Turing Award)
- PCNN+MIL-Attention 相关论文（Zeng et al. EMNLP 2015 等）
- 本项目设计文档：`docs/IMPLEMENTATION_*.md`
- Neo4j CSV 导入最佳实践

---

**使用说明**：直接运行 `python run.py` 即可体验完整构建流程。欢迎提出改进意见或共同重构项目结构，使其更加清晰、专业。

**最后更新**：2026年4月
```

**说明**：我已在项目根目录创建了全新的 `README.md`，内容完全基于你**当前代码的真实实现**撰写，比对方文档更简洁、专业、真实，同时诚实地指出了结构问题，并给出了具体改进建议。

如果你对内容有任何需要调整的地方（比如增加具体统计数据、修改语气、补充某个模块细节），请告诉我，我可以立即使用工具再次编辑。