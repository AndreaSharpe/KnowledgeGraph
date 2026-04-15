## 文档目的（必须读完）

本文档是本项目接下来一轮改进的**需求规格（SPEC）**，用于让任何后续 AI/开发者在不额外沟通的前提下，**准确理解并实现**以下目标：

- 让知识图谱从“只围绕人物图灵”升级为“同时覆盖图灵、图灵机、图灵奖”的**多核心主题图（multi-seed KG）**。
- 让实体抽取从“锚点窗口启发式”升级为更经典的**文档级抽取 + 相关性/归因**机制（仍保持可解释、可审计）。
- 在项目中体现“学习并使用了传统机器学习方法（条件随机场 CRF）做实体抽取”的能力：允许对**某一指定数据源**采用 CRF NER，其余大部分数据源仍使用当前的小模型（spaCy）抽取。

本 SPEC 同时记录了“当前代码库已经完成的改动”，避免重复实现或误判现状。

---

## 当前已完成的改动（现状基线）

以下内容已经在代码中实现（后续工作应在此基础上扩展，而非推倒重来）：

### A. 锚点窗口抽取（NER + 实体链接）

- 已在 `turing_kg/ner_link.py` 实现：以含锚点句为中心，向左右扩展 `anchor_window_sentences`（默认 1），在扩展后的句集上做 spaCy NER + Wikidata 实体链接。
- 已加入可解释字段：`LinkedSpan.context` 取值 `anchor` / `neighbor_window`，并写入边的 snippet（`ctx=...`）。
- 降噪约束：对“邻接窗口句（非锚点句）”禁用 `PERSON` 链接，仅保留 ORG/GPE/LOC 等（避免代词/旁人名造成误链）。
- 配置项已加入 `sources/ner_link_config.json`：
  - `anchor_window_sentences`

### B. 关系抽取（中文模板 / 英文依存）

- 已新增 `turing_kg/relation_patterns.py`：
  - 中文：基于触发词与正则槽位（毕业于/就读于/出生于/逝世于/任职于…）抽取关系，再对槽位做实体链接。
  - 英文：在含锚点句上用 spaCy 依存结构抽取关系（如 born in、worked/studied at/in…），再实体链接。
- 已接入流水线：`turing_kg/build.py` 会对每个文本源同时做
  - `extract_linked_spans` → `ingest_linked_spans`
  - `extract_pattern_relations` → `ingest_pattern_relations`

### C. 导出增强

- `data/triples_extracted.csv` 已新增列 `extraction_method`，用于区分：
  - `ner_entity_linking:anchor` / `ner_entity_linking:neighbor_window`
  - `zh_template` / `en_dependency`

---

## 新增/扩展需求总览（本轮必须实现）

本轮需求分三块：**多核心主题（multi-seed）**、**相关性/归因机制**、**CRF NER（单源启用）**。三者必须能协同工作，并且输出可审计。

### 1) 多核心主题实体（Multi-seed KG）

#### 1.1 目标

当前项目默认以“图灵（人物）”作为唯一根实体（root）。需要升级为至少三个并列的核心主题实体（seed entities）：

- 图灵（人物）
- 图灵机（概念/理论模型）
- 图灵奖（奖项）

每个 seed 都应该能够：

- 作为文本抽取/关系抽取的“归因主体”（该条抽取结果属于哪个 seed 子图）。
- 作为结构化层（Wikidata）邻域扩展的中心（可选，见 1.3）。

#### 1.2 约束

- **不得在代码里硬编码 QID**（容易错、不可维护）。QID 应由配置/映射文件提供。
- seeds 的锚点（中/英）必须可配置，便于后续增加“停机问题”“可判定性”等主题 seed。

#### 1.3 建议的实现形态（实现时必须保持可解释）

新增配置文件（建议路径，允许等价实现但语义必须一致）：

- `sources/seed_entities.json`

内容结构建议如下（字段名可微调，但含义必须一致）：

```json
{
  "seeds": [
    {
      "seed_id": "turing_person",
      "qid": "Q7251",
      "type": "Person",
      "anchors_zh": ["艾伦·图灵", "阿兰·图灵", "图灵"],
      "anchors_en": ["Alan Turing", "Alan Mathison Turing", "A. M. Turing", "Turing"]
    },
    {
      "seed_id": "turing_machine",
      "qid": "<由 entity_map 或人工确认填写>",
      "type": "Concept",
      "anchors_zh": ["图灵机"],
      "anchors_en": ["Turing machine"]
    },
    {
      "seed_id": "turing_award",
      "qid": "<由 entity_map 或人工确认填写>",
      "type": "Award",
      "anchors_zh": ["图灵奖"],
      "anchors_en": ["Turing Award"]
    }
  ]
}
```

并在 `sources/entity_map.csv` 中确保 “图灵机/图灵奖” 有稳定的 QID 映射（作为链接优先级/override）。

#### 1.4 验收标准

- 任一文本源中，若句子明显围绕“图灵机/图灵奖”讨论，应能在导出中看到：
  - 边的 `:START_ID` 为对应 seed 的 QID（而不是一律为 Q7251）
  - triples 中能追溯该条结果的 seed 归因（见第 2 部分导出要求）

---

### 2) 相关性建模 + 归因（Attribution）机制

#### 2.1 目标

将“锚点窗口”从唯一策略升级为**可组合的相关性/归因框架**：

- 抽取时不再只问“这句有没有图灵锚点”，而要回答：**这句最可能在讲哪个 seed？**
- 抽取结果需要带上：**seed 归因、证据句、方法、分数/理由**，保证可审计。

#### 2.2 最小可行机制（必须实现）

实现一个“句子路由（sentence routing）+ seed 归因”的模块：

- 输入：文档文本（或已切句列表）、seed 配置（anchors）、可选触发词表
- 输出：对每个句子给出
  - `assigned_seed_qid`（可为 None）
  - `assigned_seed_id`
  - `relevance_score`（0~1 或任意一致标度）
  - `reasons`（可选，但强烈建议：例如命中哪些 anchor/触发词、距离等）

路由建议采用“规则 + 线性加权”即可（不需要复杂模型）：

- anchor 命中：强证据（同语言 anchor）
- 触发词（领域词）：中证据（例如“获奖/得主/ACM/award/recipient”偏图灵奖；“可计算/机器/停机/automaton”偏图灵机）
- 距离特征：句距/段距作为弱证据（可复用当前窗口实现）

#### 2.3 与现有抽取模块的集成要求（必须明确归因）

将以下抽取结果从“默认挂到 ROOT_ENTITY_QID”改为“挂到 assigned seed”：

- NER + 实体链接结果（`extract_linked_spans`）
- 模板/依存关系结果（`extract_pattern_relations`）

注意：

- 现有锚点窗口策略仍可保留，但其“中心锚点”应基于**句子路由结果**或 seed anchors，而不是只针对图灵人物。
- 若一个句子同时命中多个 seed（例如人物介绍段落同时提到图灵机），应支持：
  - 选择最高分 seed（最简单）
  - 或多归因（同一句输出两份抽取，需去重策略）
 具体选择哪一种，必须在实现与导出中写清楚，并保持一致。

#### 2.4 导出/可审计要求（必须实现）

扩展 `data/triples_extracted.csv`（或新增一个更详细的 CSV/JSON），至少包含：

- `seed_qid`
- `seed_id`
- `relevance_score`（若实现了）
- `extraction_method`（已存在，需继续保留）
- `evidence_sentence` / `snippet`（已存在 snippet，可继续沿用）

目的：任何人拿到导出文件，能回答：

- “这条边是属于图灵机子图还是图灵奖子图？”
- “它为什么被抽出来？证据句是哪句？方法是什么？”

---

### 3) CRF（条件随机场）NER：对“某一数据源”启用

#### 3.1 目标

在项目中体现“学习使用了 CRF 等传统机器学习方法做实体抽取”：

- 允许对**某一指定数据源**使用 CRF NER（例如：`raw/excerpts/book_excerpt.txt` 或某个 `raw/excerpts/articles/*.txt` 或某个 PDF 章节文本）。
- 其余大部分数据源继续使用当前的小模型抽取（spaCy），保证工程复杂度可控。

#### 3.2 关键约束（避免被误解）

- CRF NER **不是替换整个系统**，而是“可选后端/可选策略”，只在指定 source 上启用。
- CRF 的输出格式必须与 spaCy NER 的 downstream 对齐：最终仍要输出实体 mention 与类型（PER/ORG/GPE/LOC 等），并能走同一套实体链接与关系抽取（或至少能走实体链接）。
- 允许“效果一般”，但需要**机制完整**：包含数据格式、特征、训练、评估、推理、集成点与可复现命令。

#### 3.3 数据与训练要求（最小可行）

新增一个 CRF 标注数据目录（建议路径）：

- `data/ner_crf/`
  - `train.conll`
  - `dev.conll`
  - `test.conll`（可选）

采用经典 CoNLL 格式：

- 每行：`TOKEN<TAB>LABEL`
- 句子之间空行分隔
- 标签体系：BIO 或 BIOES（必须明确选择一种，并在代码中固定）

中文建议 token 粒度为“字级”（降低分词依赖），英文为“词级”。

训练脚本建议新增：

- `scripts/train_crf_ner.py`

输出模型文件：

- `models/crf_ner.pkl`（或同等格式）

评估建议输出：

- token-level accuracy
- entity-level F1（可用 `seqeval`）

#### 3.4 推理与工程集成（必须实现）

新增 CRF NER 推理模块（建议路径）：

- `turing_kg/crf_ner.py`

接口建议与现有 NER 抽取对齐，例如：

- `extract_entities_with_crf(text, model_path, lang) -> list[EntitySpan]`

并在文本源处理环节增加“按 source 选择 NER backend”的机制（建议配置）：

- 在 `sources/ner_link_config.json` 或新增 `sources/extraction_profile.json` 中声明：
  - 哪些 source 使用 `crf`，哪些使用 `spacy`

示例（语义必须一致）：

```json
{
  "default_ner_backend": "spacy",
  "per_source_overrides": {
    "raw/excerpts/book_excerpt.txt": "crf"
  }
}
```

#### 3.5 验收标准

- 能通过命令完整跑通一次：
  - 训练 CRF 模型（生成模型文件）
  - 用 CRF 模型对指定 source 抽取实体
  - 抽取结果进入实体链接与图构建，导出可看到 `extraction_method` 标注为 `crf`（或 `crf_ner`）
- 其余 source 仍用 spaCy，且不需要额外标注数据即可运行。

---

## 非目标（明确不做，防止 AI 过度扩展）

- 不要求用大语言模型完成 NER/RE（可以完全不引入 LLM）。
- 不要求实现复杂的全局消歧优化、端到端训练、或大规模标注。
- 不要求把所有关系都覆盖到教科书的“事件抽取/事件关系/过程抽取”等完整谱系；本轮重点是多主题结构 + 归因 + CRF 体现。

---

## 实现建议的文件改动清单（供后续执行时对照）

以下是建议改动点，允许等价实现，但语义必须一致：

- 新增：
  - `sources/seed_entities.json`
  - `turing_kg/seed_router.py`（句子路由/归因 + 相关性打分）
  - `turing_kg/crf_ner.py`
  - `scripts/train_crf_ner.py`
  - `data/ner_crf/*`（标注数据）
  - `models/`（存模型文件，注意不要提交大文件到 git，或改为只提交说明）
- 修改：
  - `turing_kg/build.py`：支持多 seed、按 seed 归因入图、按 source 选择 NER backend
  - `turing_kg/ner_link.py`：从“只给 root”改为“给指定 seed”；锚点窗口应基于 seed anchors/路由
  - `turing_kg/relation_patterns.py`：同样需要 seed-aware（按 assigned seed 入图）
  - `turing_kg/export_io.py`：导出包含 seed 归因字段（CSV/JSON）
  - `sources/ner_link_config.json`：加入 backend 选择、CRF 模型路径等配置项（或新建配置文件）

---

## 关键设计原则（强制）

- **可解释**：每条抽取结果必须能说明“为什么属于这个 seed（证据/理由/分数）”。  
- **可配置**：seed 列表、锚点、后端选择、窗口大小必须在配置中可改。  
- **低耦合**：CRF 只影响指定 source，不影响全局可运行性。  
- **可复现**：CRF 训练与推理必须有确定命令与固定输入输出路径。  

