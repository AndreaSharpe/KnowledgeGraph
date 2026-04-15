## 目标与约束

本设计文档用于落地 `docs/SPEC_multi_seed_relevance_crf_ner.md` 的三项升级：

- 多核心主题（multi-seed）：图灵（人物）、图灵机（概念）、图灵奖（奖项）
- 相关性建模 + 归因（sentence routing / attribution）：抽取结果必须可审计（seed、分数、理由、证据句）
- CRF NER：仅对指定 PDF（`raw/艾伦·图灵传：如谜的解谜者.pdf`）的 1–50 页启用；其余 source 继续 spaCy

强制约束（与 SPEC 一致）：

- 不硬编码 QID：seed 的 QID 只能来自配置（或复用现有 entity_map 覆盖机制）
- 可解释：每条抽取结果能说明“为什么属于这个 seed”（理由/分数/证据句）
- 低耦合：CRF 仅影响指定 source，不影响全局可运行性
- 可复现：CRF 训练与推理路径固定、可一键运行

本文档只做“架构与接口设计 + 新文件骨架”。不会修改现有代码。

---

## 现状基线（用于复用）

你当前代码已经具备：

- 文本采集：`turing_kg/text_sources.collect_text_sources()`
  - Wikipedia 摘要：当前只抓取图灵人物（通过 `fetch_turing_excerpts()`）
  - articles：`raw/excerpts/articles/*.txt`
  - book_excerpt：`raw/excerpts/book_excerpt.txt`
  - PDF：读取 `sources/pdf_sources.json`，并通过 `pdf_text.extract_pdf_pages()` 抽页段
- 抽取：
  - spaCy NER + Wikidata 实体链接：`turing_kg/ner_link.extract_linked_spans()`
  - 规则关系抽取：`turing_kg/relation_patterns.*`
- 导出：
  - `data/triples_extracted.csv`：已有 `extraction_method` 列，NER 行包含 `ner_label`
- 实体链接：`turing_kg/entity_linking.link_mention_to_qid()`（含 entity_map override）

本轮设计的核心是：在不推倒现有函数的前提下，引入“seed 概念”与“路由/归因层”，并在 build 流水线里把 subject 从单 root 改为（可能多个）seed。

---

## 顶层数据流（推荐）

对每个文本源（Wiki / article / book / pdf）：

1. **切句**（复用现有 `_split_sentences` 逻辑，后续可抽成公共函数）
2. **句子路由**：对每句计算它与各 seed 的相关性，输出 top-k 归因
3. **按归因分组抽取**：
   - 对每个 seed，取归因到它的句子集合（可带邻接窗口），作为抽取输入
4. **NER 后端选择**（按 source 配置）：
   - default：spaCy（沿用 `extract_linked_spans`，但需要“seed-aware anchors”）
   - override：CRF（仅 PDF 1–50 页 source）
5. **实体链接**：对 NER mention 使用 `link_mention_to_qid`（完全复用）
6. **关系抽取**：复用现有 `extract_pattern_relations`，但输入句子集合按 seed 划分
7. **入图与导出**：每条结果携带 seed 与 attribution 字段（score、reasons、evidence_sentence）

多归因（你的选择）意味着：同一句可能同时归因多个 seed，因此**同一句抽取结果可能会复制多份**，分别挂到不同 seed 的 subject 上。需要去重策略（见后文）。

---

## 配置设计（新建文件，不改现有配置也能并存）

### 1) `sources/seed_entities.json`

定义 seeds 的唯一来源：seed_id、qid、锚点、触发词、类型等。QID 允许暂时留空，但运行时必须可解析（通过 entity_map 或 Wikidata 搜索）。

关键字段建议：

- `seed_id`: 稳定的内部 ID（如 `turing_person`）
- `qid`: Wikidata QID（不硬编码在代码里，只存在配置里）
- `anchors_zh`: 中文锚点列表（用于路由强证据）
- `triggers_zh`: 触发词列表（用于路由中证据）
- `type`: 可选，用于审计

### 2) `sources/attribution_config.json`

路由与多归因的控制开关：

- `top_k`: 每句输出 top-k seeds（例如 2）
- `min_score`: 低于该分数不归因
- `window_sentences`: 对归因句做邻接扩展（类似现有 anchor window）
- `score_weights`: anchor/trigger/distance 的线性权重

### 3) `sources/extraction_profile.json`

声明 per-source 的 NER backend：

- `default_ner_backend`: `spacy`
- `per_source_overrides`:
  - `pdf:file:...#pages=1-50` 或 `raw/艾伦·图灵传：如谜的解谜者.pdf#pages=1-50` → `crf`
- `crf_model_path`: `models/crf_ner.pkl`

说明：由于现有 `collect_text_sources()` 对 PDF 的 `source_url` 形如 `file:<path>#pages=...`，建议 override key 使用同样形式，避免歧义。

---

## 代码模块设计（新建骨架）

### A) `turing_kg/seed_config.py`

职责：加载 seeds 配置与 attribution/extraction 配置，提供数据类：

- `SeedSpec(seed_id, qid, anchors_zh, anchors_en, triggers_zh, triggers_en, type)`
- `AttributionConfig(top_k, min_score, window_sentences, weights, ...)`
- `ExtractionProfile(default_ner_backend, per_source_overrides, crf_model_path, ...)`

### B) `turing_kg/seed_router.py`

职责：句子路由（相关性 + 归因），输出可审计理由。

输入：

- `sentences: list[str]`
- `seeds: list[SeedSpec]`
- `cfg: AttributionConfig`

输出：

- 每句一个 `SentenceAttribution`：
  - `sentence`
  - `assigned: list[SeedAttribution]`（按分数降序，最多 top_k）
  - `debug`（可选）

打分建议（线性规则，足够验收且易解释）：

- anchor 命中：每命中一个 anchor 加 `w_anchor`，并记录命中的 anchor
- trigger 命中：每命中一个 trigger 加 `w_trigger`
- distance：若同段/相邻句出现 anchor，可给弱加分（用于 PDF 段落连续性）

`reasons` 字段建议结构化而非自然语言，以便导出/审计：

```json
{
  "anchor_hits": ["图灵奖"],
  "trigger_hits": ["得主", "ACM"],
  "distance_bonus": 0.1
}
```

### C) `turing_kg/ner_backend.py`

职责：统一 NER 后端接口，屏蔽 spaCy / CRF 差异。

建议接口：

- `extract_entities(text_or_sentences, backend, lang, ...) -> list[EntitySpan]`

其中 `EntitySpan` 至少包含：

- `mention`
- `label`（七类之一）
- `sentence` / `snippet`（证据句）
- `start/end`（可选，用于调试）

### D) `turing_kg/crf_ner.py`

职责：实现 CRF 模型的推理与（可选）训练辅助；训练主入口放在脚本中。

关键点：

- 中文 token 粒度：默认字级（可加简单规则合并连续数字/百分号）
- 特征：char ngram、is_digit、is_hanzi、is_punct、shape 等传统特征
- 标签：BIO + 7 类（PER/ORG/LOC/TIME/DATE/MONEY/PERCENT）

### E) `scripts/train_crf_ner.py`

职责：训练与评估可复现，输出模型文件。

固定输入输出：

- 输入：`data/ner_crf/train.conll`、`data/ner_crf/dev.conll`（可选 test）
- 输出：`models/crf_ner.pkl`
- 评估：token accuracy、entity F1（使用 `seqeval`）

依赖建议：

- `sklearn-crfsuite`
- `seqeval`

### F) 导出扩展（设计先行）

由于你要求本轮“不改现有代码”，这里先给出目标字段定义，后续接入时实现：

在 `triples_extracted.csv` 增加列（建议）：

- `seed_id`
- `seed_qid`
- `relevance_score`
- `reasons`（可截断）
- `evidence_sentence`（或复用 snippet）

同时建议新增 debug 导出（可选）：

- `data/sentence_attribution.jsonl`：逐句记录归因结果

---

## 多归因去重策略（必须有）

由于同一句可归因多个 seed，抽取结果复制会导致重复边。推荐的去重键：

- NER 边：`(seed_qid, object_qid, predicate, evidence_sentence_hash)`
- 关系边：`(seed_qid, predicate, object_qid, evidence_sentence_hash)`

若重复出现：

- 合并 `reasons`（并集）
- 取最大 `relevance_score` 或做加权平均

---

## “复用抓取函数”的实现建议（不改现有代码的前提下）

你提到“从维基百科 API 上抓取图灵机和图灵奖摘要、以及 Wikidata 结构化数据复用之前写的”。推荐做法：

- 新增一个通用函数：`fetch_wikipedia_excerpts_for_seeds(seeds)`，内部复用你现有 `wikipedia_text` 的请求逻辑（当前只有 `fetch_turing_excerpts`，后续可扩展为更通用）
- 结构化层：让 `load_structured_graph` 支持传入多个 seed（QID 列表）并合并结果（仍复用现有 Wikidata 查询封装）

注意：这些属于“后续改现有代码时”的接入点，本轮只先设计与新增骨架文件。

---

## 你已确定的 PDF 范围（用于配置）

PDF 抽取页段：**1–50 页**。

建议在 `sources/pdf_sources.json` 中为该 PDF 配置：

- `enabled: true`
- `page_start: 1`
- `page_end: 50`

（本轮不修改该文件；后续接入时按此参数配置即可。）

---

## 分阶段落地路径（后续接入时按顺序执行）

阶段 0：配置与骨架（本轮完成）

- 新增 `seed_entities.json`、`attribution_config.json`、`extraction_profile.json`
- 新增 `seed_config.py`、`seed_router.py`、`crf_ner.py`、`ner_backend.py`
- 新增训练脚本 `scripts/train_crf_ner.py`
- 新增 `data/ner_crf/README.md`（说明数据格式与标注规范）

阶段 1：多 seed 结构化层（改现有代码）

- 扩展 structured graph：多 seed 的 Wikidata 邻域合并

阶段 2：句子路由与 seed-aware 抽取（改现有代码）

- build 流水线引入路由
- NER/关系抽取按 seed 分组入图
- triples 导出补齐 seed 字段

阶段 3：CRF 训练与单源启用（改现有代码 + 训练）

- 准备 `data/ner_crf/train/dev` 标注
- 训练生成 `models/crf_ner.pkl`
- extraction_profile 对 PDF 1–50 页启用 CRF

阶段 4：审计与验收

- 通过导出文件验证三子图均有边
- 抽样检查 reasons 与 evidence_sentence 可追溯

