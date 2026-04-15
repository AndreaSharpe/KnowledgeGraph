# 项目进度汇报：你需要掌握的内容与实现说明（实体消歧 · 关系抽取）

本文档面向**向老师口头汇报**：说明项目当前做到哪一步、各模块如何衔接，并对**实体消歧（实体链接）**与**关系抽取**两部分给出可复述的实现细节。对应代码主要在 `turing_kg/linking/`、`turing_kg/extraction/`、`turing_kg/relation/` 与 `turing_kg/build.py`。

---

## 一、汇报前建议掌握的「清单」

在汇报中，老师通常会关心下面几类问题。建议你对每一项都能用**一两句话 + 一个例子**说明。

| 维度 | 你需要能说清楚的内容 |
|------|----------------------|
| **目标** | 从多源文本与 Wikidata 结构化信息构建「图灵」主题知识图谱；文本侧通过 NER+链接得到实体 QID，再抽取关系或共现。 |
| **入口与模式** | `python run.py`；`sources/build_config.json` 中 `mode`：`full`（完整 NER+链接）、`export_only`（只读已有 CSV 再导出）、`from_curated`（用已落盘的中间层重放，不调 NER/API）。 |
| **实体消歧** | 候选来自 Wikidata `wbsearchentities`；本地用标签/描述与上下文重排序；可选**协同链接**：用语义邻居（P31 等）在句子窗口内联合选 QID。 |
| **关系抽取** | **两条线**：(1) 规则/依存模板，直接产出带 `Pxx` 的边；(2) **远程监督 + PCNN + MIL-Attention**，按「实体对 bag」多标签分类，推理结果经阈值过滤后写入图或导出。 |
| **数据落盘** | 中间层：`data/processed/sentences.jsonl`、`routing.jsonl`；实体链接：`data/curated/mentions.jsonl`、`candidates.jsonl`、`resolved.jsonl`；关系 MIL：`data/curated/bags.jsonl`、`ds_labels.jsonl`、`re_predictions*.jsonl`；导出：`data/nodes.csv`、`relationships.csv`、`triples_extracted.csv` 等。 |
| **局限与诚实表述** | 链接依赖 Wikidata 搜索质量与 API 限流；协同解码是窗口内贪心迭代，非全局最优；PCNN 为字符级、数据量依赖远程监督噪声；阈值见 `sources/relation_thresholds.json`。 |

---

## 二、整体流水线（便于汇报时画一张图）

`build.py` 中的教科书式流水线可概括为：

1. **结构化层**：对每个种子实体的 Wikidata 陈述构图（`load_structured_graph_for_seeds`）。
2. **文本采集**：维基摘要、文章节选、PDF 等（`text_sources`）。
3. **句子级处理**：分句落盘；可选多种子句归属/相关性路由（`seed_router`，归因配置见 `sources/attribution_config.json`）。
4. **NER + 实体链接**：在**锚点窗口**内做 spaCy NER（及数值规则补强），对每个 mention 调链接；可选协同消歧后写 `resolved.jsonl`。
5. **关系抽取（规则）**：中文正则槽位、英文依存，槽位同样走链接（`relation_patterns`）。
6. **导出**：图 CSV、三元组 CSV；若存在 MIL 预测可合并进导出（`relation/mil_ingest`）。

关系学习部分（训练/推理）一般由脚本完成，例如 `scripts/build_relation_bags.py`、`scripts/train_relation_pcnn.py`、`scripts/infer_relation_pcnn.py`，与主构建流水线**解耦**，通过 `bags.jsonl` / checkpoint / `re_predictions.jsonl` 衔接。

---

## 三、实体消歧（实体链接）— 详细说明

### 3.1 问题定义

对文本中的每个 **mention**（字符串），在 Wikidata 上选定一个 **QID**，使该 QID 与当前语境一致。项目中的「消歧」包含两层：

- **局部（local）**：仅根据 mention、句子上下文、候选标签/描述打分。
- **全局/协同（global）**：在同一**句子窗口**内，多个 mention 的 QID 选择互相影响，通过知识库中的一跳邻居重叠度（coherence）修正选择。

### 3.2 Mention 从哪来（NER 与窗口）

实现见 `turing_kg/extraction/ner_link.py`：

- 使用 **spaCy**（`en_core_web_sm` / `zh_core_web_sm`），将英文 `PERSON/ORG/GPE/...` 等映射为项目七类标签（PER/ORG/LOC/DATE/TIME/MONEY/PERCENT）。
- 对中文日期、时间、金额、百分比等用**正则规则**补强（spaCy 易漏的数值类）。
- **锚点机制**：只在「出现种子锚点（如「图灵」「Alan Turing」）的句子」及其左右邻句（`anchor_window_sentences`，默认见 `sources/ner_link_config.json`）上做 NER+链接，控制噪声与 API 调用量。

初次链接时，对每个 span 调用 `link_mention_to_qid`（内部同样走搜索+打分），得到 `LinkedSpan.object_qid` 与分数。

### 3.3 候选生成与局部重排序（核心公式）

实现见 `turing_kg/linking/entity_linking.py`：

1. **候选**：对每个 mention，用 Wikidata API `wbsearchentities` 在中英各搜一次（`wb_search_entities`），结果去重；请求带**本地缓存**与 **429 退避**，减轻限流。
2. **打分（`rank_candidates` / `score_candidates`）**：对候选 \(i\) 综合  
   - 标签与 mention 是否精确匹配、子串匹配（label bonus）；  
   - 上下文与「label + description」的 **Jaccard** 词重叠；  
   - mention 与 label 的 **字符 3-gram Jaccard**（对拼写变体更稳）；  
   - **搜索排名先验**（越靠前的候选略加分）。  
3. **阈值**：若最高分低于 `min_link_score`（`ner_link_config.json`），且不是「仅一个候选」的兜底情况，则**拒绝链接**（返回空），避免低置信 QID 污染图谱。
4. **人工表覆盖**：`sources/entity_map.csv` 可在 `link_mention_with_candidates` 中**优先命中**（确定性映射），用于易错专名。

构建阶段为可审计性会调用 `link_mention_with_candidates`，写出完整候选列表与分项分数（`candidates.jsonl` / `resolved.jsonl` 中的 `reasons`）。

### 3.4 协同链接（Collective EL）— 消歧的「第二遍」

配置见 `sources/entity_linking_config.json`（字段会合并进 `EntityLinkingConfig`），实现见 `turing_kg/linking/collective_linking.py`，调用点在 `build.py`。

**直觉**：同一窗口内若两个 mention 分别指向「剑桥大学」与「英国」，它们在 Wikidata 上通过 P17/P131 等关系形成的**一跳邻居集合**应有交集；若选错成不相关实体，邻居交集往往较低。

**步骤概要**：

1. 对每个 mention 保留 top-K 局部候选（`collective_top_k_candidates`），构成 `MentionNode`。
2. 收集所有候选 QID，用 `wbgetentities` 拉取**声明（claims）**，仅保留指定属性边上的对象，形成 `neighbor_sets[qid]`（属性列表默认含 P31「实例」、P279「子类」、P17「国家」等，见配置 `coherence_props`）。
3. 定义两个 QID 的 coherence 为两邻居集合的 **Jaccard 相似度**。
4. 在每个**中心句索引**上取窗口 `[center ± window_sentences]` 内的 mentions；若窗口内少于 2 个 mention 则跳过（无法协同）。
5. **解码**：`collective_decode_window` 使用**坐标上升式贪心**（多轮迭代）：初始化每人选局部最高分候选；然后轮流对每个 mention，在候选集合中选使  
   `局部分数 + λ × Σ coherence(当前候选, 他人已选 QID)`  
   最大的 QID；迭代直至不变或达到 `max_iters`。λ 即 `lambda_coherence`。

**结果写回**：若解码改变了 QID，会更新 `resolved.jsonl` 中的 `chosen_qid`、`scores.global/total`，并把 `LinkedSpan.object_qid` 改为新 QID，从而影响后续入图与三元组。

**汇报时可说明的局限**：协同部分是**窗口内近似**，不是全篇联合推理；邻居仅一跳、且属性子集可配置；API 与 `max_entities_to_fetch` 限制可能截断候选邻居信息。

---

## 四、关系抽取 — 详细说明

项目里关系抽取可分为 **轻量可解释的规则/依存路径** 与 **神经远程监督（PCNN + MIL）** 两条路径，二者互补。

### 4.1 基于模板与依存的关系抽取（与链接共用基础设施）

实现见 `turing_kg/extraction/relation_patterns.py`：

- **中文**：在含锚点的句子上，用若干**正则模板**捕获槽位（如「毕业于 X」「出生于 X」），对槽位字符串调用 `link_mention_to_qid`，成功则输出带 **Wikidata 属性 id**（如 P69、P19）的 `PatternRelation`，再写入图。
- **英文**：在含图灵锚点的句子上，用 spaCy **依存句法**（如 nsubj / prep / pobj）抽取「主语—谓词—宾语」式关系，同样对宾语槽做链接。

特点：**可解释、冷启动友好**；覆盖面受模板与句法限制。

### 4.2 远程监督与 Bag 构造（MIL 的数据准备）

**Bag 定义**：固定一对 Wikidata 实体 `(subject_qid, object_qid)`（通常 subject 为种子人物/概念，object 为某次链接得到的客体），在同一文档内所有「同时出现这对实体指称」的句子（实例）组成一个 bag —— 典型 **多实例学习（MIL）** 设定：关系标签标注在 bag 上，句子中只有部分是真证据句。

实现见 `turing_kg/relation/bags.py`：

- 输入：`data/processed/sentences.jsonl`、`data/processed/routing.jsonl`、`data/curated/resolved.jsonl`。
- 过滤：仅 `chosen_qid` 为有效 QID 且**非自环**（object ≠ seed）；句子 **zh_ratio** 不低于门控（默认 0.3），与实现文档一致。
- 输出：`data/curated/bags.jsonl`，每条记录含 `bag_id`、`subject_qid`、`object_qid`、`instances`（多条句实例及 mention 信息）等。

**远程监督标签**（`turing_kg/relation/ds_labels.py`）：对每个 bag，查询 Wikidata 上是否确实存在边 `(subject_qid, prop_id, object_qid)`，且 `prop_id` 在该种子类型允许的关系空间内（`sources/relation_allowlist.json`）。若存在，则 `prop_id` 作为该 bag 的正标签之一（多标签）。标签噪声来自：KB 边与文中语义未必对齐、或同对实体多种关系并存。

### 4.3 模型：PCNN + MIL-Attention

实现见 `turing_kg/relation/pcnn_mil.py`：

- **PCNN**：字符序列 → Embedding → 一维卷积 → 按两个实体位置将卷积特征分为**三段 piecewise max pooling** → 拼接 → 全连接得到句级特征；实体位置来自句子中种子锚点与客体 mention 的字符跨度（由 `bag_to_model_batch` 等逻辑确定，见 `pcnn_train.py`）。
- **MIL-Attention**（Lin et al., 2016 风格）：同一 bag 内多个句向量 `{h_i}` 经共享注意力加权求和得到 bag 表示 `h_bag`，再经同一分类头得到 **多标签 logits**。
- **训练**（`pcnn_train.py`）：对 bag 级多标签使用 **BCE with logits**，并按类别频次设置 **pos_weight** 缓解类别不平衡。

### 4.4 推理与入库

实现见 `turing_kg/relation/pcnn_infer.py`：

- 加载 `models/relation_pcnn/pcnn_{seed_type}.pt` 与元数据（字符表、关系空间、超参）。
- 对每个 bag 前向得到各类别概率；**注意力权重**用于选出「最支持该 bag 分类」的句子，写入 `evidence`（便于审计与展示）。
- 使用 `sources/relation_thresholds.json` 中**按属性**的阈值过滤；通过的预测可经 `mil_ingest.ingest_mil_edges_to_graph` 写入图（`provenance=mil_relation_extraction`），或合并进导出三元组。

**汇报时可说明**：推理按 `seed_type`（如 Person / Concept / Award）分模型；`infer_all_available` 会对存在 checkpoint 的类型依次推理并合并为 `re_predictions.jsonl`。

---

## 五、配置文件与脚本速查（汇报时可一带而过）

| 路径 | 作用 |
|------|------|
| `sources/build_config.json` | 构建模式 `full` / `export_only` / `from_curated`。 |
| `sources/ner_link_config.json` | 锚点文本、`min_link_score`、锚点窗口句数。 |
| `sources/entity_linking_config.json` | 协同链接开关、窗口、λ、top-K、coherence 属性、拉取实体上限。 |
| `sources/entity_map.csv` | 高置信人工映射，优先于搜索。 |
| `sources/relation_allowlist.json` | 各种子类型可用的 Wikidata 属性集合（远程监督与导出空间）。 |
| `sources/relation_thresholds.json` | MIL 推理按属性的概率阈值。 |
| `scripts/build_relation_bags.py` | 生成 `bags.jsonl`。 |
| `scripts/train_relation_pcnn.py` / `train_relation_pcnn.ps1` | 训练 PCNN-MIL。 |
| `scripts/infer_relation_pcnn.py` | 推理并写 `re_predictions*.jsonl`。 |

---

## 六、汇报结构建议（3–5 分钟口述骨架）

1. **背景与目标**：图灵主题 KG，多源文本 + Wikidata。  
2. **实体侧**：NER → Wikidata 候选搜索 → 局部打分 →（可选）窗口内协同解码；产出 `resolved.jsonl`。  
3. **关系侧**：规则/依存 + 远程监督 bag + PCNN-MIL，阈值过滤与证据句注意力。  
4. **当前进度**：已跑通链路（据你实际环境说明：例如是否已训练 checkpoint、导出行数、典型问题）。  
5. **局限与下一步**：API 与噪声、模板覆盖、模型与数据规模、评价指标（若有）。

---

*文档生成自仓库当前实现；若代码变更，请以 `turing_kg/build.py` 与各子模块为准。*
