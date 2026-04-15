## 文档目的

给 `KnowledgeGraph` 项目补齐并“尽可能完美”地实现**非结构文本关系抽取（RE）**能力：在多中心（multi-seed）知识图谱场景下，统一关系 schema、可审计的训练/推理数据层、以及高质量的 **远程监督（DS）+ 多实例学习（MIL-Attention）** 关系分类模块，并与现有规则/模板抽取融合，最终输出可入库、可解释、可评估的三元组。

本文既是工程实现路径，也是验收标准与研究路线图。

---

## 0. 当前代码现状（基线能力与约束）

项目已具备以下能力（本方案在此基础上扩展，而不是推倒重来）：

- **多中心 seeds**：`sources/seed_entities.json` 定义了 `turing_person / turing_machine / turing_award`，并由 `turing_kg/attribution/seed_router.py` 做句子路由/归因（top-k + 阈值 + 可选窗口扩展），`turing_kg/build.py` 按 seed 循环执行抽取并入图。
- **实体识别 + 实体链接（NER/EL）**：`turing_kg/extraction/ner_link.py` 与 `turing_kg/linking/*` 已实现候选生成、局部打分与可选协同一致性解码（Collective EL），并能把决策审计写入 `data/curated/*.jsonl`。
- **关系抽取（规则/模板）已存在**：`turing_kg/extraction/relation_patterns.py` 实现了中文模板（正则槽位）+ 英文依存句法抽取，并在 `build.py` 中写入图与 `triples_extracted.csv`。

关键约束：

- **多中心（multi-seed）**：图中存在多个 root（Q7251/Q163310/Q185667），抽取需能“按中心归因”，并避免不同中心之间的关系空间/类型冲突。
- **工程化优先**：要可复现（配置化）、可审计（证据句+理由+置信度）、可评估（gold dev/test + 指标）。

### 0.1 可移植性要求（本方案的硬约束）

为了确保后续能“便捷移植到其他关键主体/其他 seeds”，本方案要求：

- **任何主体特化（例如“图灵”字符串、固定 QID）不得写死在模型/抽取器逻辑中**，必须来自 `sources/seed_entities.json` 与关系 schema 配置。
- **关系集合、类型约束、语言策略、阈值、训练超参全部配置化**，以便替换主体时做到“改配置不改代码”。
- **所有中间层输出（bags/ds_labels/predictions）必须使用稳定键**：至少包含 `source_id + seed_id/subject_qid + object_qid + sentence_idx`，否则跨主体迁移时无法对照与复跑。

注意：当前 `turing_kg/extraction/relation_patterns.py` 的英文依存抽取存在明显的“主体特化”（函数名与实现以 Turing 为锚点）。后续若要移植到其他主体，必须把该部分抽象为“seed-aware 的 anchor/head 识别”，详见第 7 节的工程落地建议与第 10 节的移植清单。

---

## 1. 总体目标与边界

### 1.1 目标输出（你最终要“用”的结果）

对于每个 seed（中心实体）与文本源，关系抽取模块输出并落盘：

- 三元组：`(subject_qid, predicate_prop_id, object_qid)`（优先对齐 Wikidata `Pxxx`）
- 证据：`source_id/source_url/citation_key/sentence_idx/evidence_sentence/snippet`
- 置信度：`score`（可校准）、`extraction_method`
- 可解释：MIL Attention 的 top-k 证据句（权重）、或规则命中说明

并最终：

- 入图：写入 `relationships.csv`（或 Neo4j 导入结构），`provenance` 区分 `pattern_relation_extraction` vs `mil_relation_extraction`
- 导出：`data/triples_extracted.csv` 增量字段不破坏兼容

### 1.2 范围边界：优先做“限定域（Schema-based）RE”，开放域后置

- **第一优先级**：限定域关系分类/抽取（关系集合固定，可对齐到 Wikidata 属性），满足“入图可用 + 可评估”。
- **开放域 OpenIE**：仅作为“候选发现/扩 schema”手段后置，不作为第一阶段主干（否则对齐困难、噪声大、难以验收）。

### 1.3 语言策略（本项目建议：中文优先，避免中英混合拖累质量）

背景：在多语语料混合时，若实体链接阶段没有把中英文 mention 很好地归一到同一个 QID，会导致：

- bag 被错误拆分（中文对象一个 QID、英文对象另一个 QID 或空 QID）
- DS 标签匹配失败（因为 object_qid 对不上）
- RE 模型学到“语言差异”而不是“关系证据”

因此在本方案中，**关系抽取优先采用中文主干闭环**，并把“中英混合对齐问题”作为 EL/归一化的后续专项优化，而不是在 RE 阶段硬抗。

落地原则：

- **训练/评估主语料**：仅选中文为主的 sources（或中文占比高的段落/句子），先把 schema、DS、MIL、gold 评估闭环做稳。
- **实体统一的唯一标准仍然是 QID**：即便文本是中文，最终 object 仍链接到 Wikidata QID，这样图层面天然跨语言统一（不依赖 label 语言对齐）。
- **对“中文句子夹杂英文专名”的处理**：允许 object mention 是英文，但在中文句子中出现时仍可用 `languages=("zh","en")` 的 EL 做 QID 归一；只要 QID 一致，RE 不会因为表面语言不同而断链。
- **英文语料的 RE**：第一阶段可暂时关闭或仅保留规则抽取（若能高精），待 EL 的跨语言归一能力提升后再纳入 MIL 训练。

---

## 2. 多中心场景：关系集合该怎么设计（统一 + 分中心约束）

### 2.1 结论：关系 schema 统一定义，但按中心（seed type）设 allowlist + 类型约束

推荐采用“两层结构”：

1) **全局统一关系表（Global Relation Registry）**
- 每条关系对齐到 `prop_id = Pxxx`
- 给出 `label`、定义、以及 subject/object 类型约束（软/硬）

2) **按 seed_type 的 allowlist（中心子集）**
- Person/Concept/Award 各自允许的 `Pxxx` 子集不同
- 抽取器/模型只在 allowlist 内预测（训练与推理都做 mask），显著降噪

这样既保持全项目的统一导出/评估口径，又避免“图灵机/图灵奖”被 person 关系模板污染。

### 2.2 需要新增的配置文件（建议）

建议新增：

- `sources/relation_schema.json`：全局关系表（Pxxx + 约束 + 描述 + 触发词可选）
- `sources/relation_allowlist.json`：按 `seed_type`（或 `seed_id`）的关系子集配置

注意：allowlist 既可以按 `seed_type`（更通用），也可以按 `seed_id`（更精细）。优先按 `seed_type`，必要时对特定 seed 覆盖。

### 2.3 “关系集合太少够不够？”：必须分阶段扩容（否则 DS 噪声与工程成本会爆炸）

结论：

- **够不够取决于你的验收目标**：如果目标是“图里边可用、可审计、可评估”，第一阶段关系不应无限扩；应先覆盖“高价值 + 高一致性 + 易类型约束”的关系，建立稳定闭环与 gold 评估，再逐步扩容。
- **图灵相关 KG 最容易失控的地方**在于：人物/机构/作品/事件/概念混在同一语料里，若你一次性把几十上百个 `Pxxx` 都纳入模型输出空间，训练会严重长尾、负例定义困难、噪声难控，最后“看似关系多，但正确率很低”，反而不如少而精。

因此建议采用如下扩容策略（每阶段都能验收）：

- **Phase A（强基线，先稳住精度）**：每个 `seed_type` 选 10–20 个关系，全部能对齐 `Pxxx`、能做类型约束、且在你的语料中出现概率高。
- **Phase B（扩覆盖，仍可控）**：每个 `seed_type` 再加 10–30 个关系，但要求：至少满足“可类型约束”或“可通过规则/触发词强过滤”，并且必须有 dev 上的阈值校准。
- **Phase C（研究/加分项）**：引入开放域候选发现（OpenIE/LLM/弱监督模板归纳），但产物进入“候选池”，不直接入主图，除非完成属性对齐与评估。

外部参考（用于“关系上界”与“关系类型画像”）：

- 图灵人物的结构化关系（可视作 Person 关系空间上界）：Wikidata `Q7251`（Alan Turing）`https://www.wikidata.org/wiki/Q7251`
- 图灵奖条目的结构化关系（可视作 Award 关系空间上界）：Wikidata `Q185667`（Turing Award）`https://www.wikidata.org/wiki/Q185667`
- 图灵奖的文本字段画像（例如 awarded for / presented by / reward / first award 等，提示哪些关系在自然语言中常被表述）：Wikipedia `Turing Award` `https://en.wikipedia.org/wiki/Turing_Award`

---

## 3. 数据层：DS + MIL 的核心是“bag 数据构造”

MIL-Attention 的成败主要取决于 bag 构造与噪声控制。你项目已有 `sentences.jsonl / routing.jsonl / resolved.jsonl`，非常适合扩展出可审计的 RE 数据层。

### 3.1 统一的 bag 定义

对每个中心 seed（subject）：

- **subject_qid**：当前 seed 的 QID（root）
- **object_qid**：文本中出现的候选实体（来自 EL 的最终 `chosen_qid`）
- **bag**：该 subject-object 对在同一 source 中出现的多句证据集合

建议 bag key：

- `bag_id = hash(source_id, subject_qid, object_qid)`（或直接用三元组字符串）

bag 内元素（instances）建议包含：

- `sentence_idx`
- `sentence_text`
- `entity_spans`（可选：subject/object 的 mention 与 char span；如暂时没有 char span，可先用“句子内是否包含 anchor/mention”做弱定位）
- `seed_attribution`（来自 routing：score/reasons，用于后续过滤/加权/审计）

### 3.2 句子范围：同句优先，其次窗口扩展（与现有门控兼容）

为了质量，建议分两级 bag：

- **同句 bag（high-precision bag）**：只收“同一句同时出现 subject anchor 与 object mention”的句子
- **窗口 bag（recall bag）**：允许在 `window_sentences` 内 co-occur（由 routing 扩展策略控制），但需要更严格的负例/阈值与类型约束

第一阶段建议只用同句 bag 做模型训练与评估；窗口 bag 作为后续提升召回的阶段。

### 3.3 DS 标签生成（远程监督）

DS 正例来源：

- 从 Wikidata 结构化层获得 `seed` 的已知关系：存在 `(subject_qid, Pxxx, object_qid)` 则认为该 bag 的关系包含 `Pxxx`。

重要注意：

- “KB 未出现”不等于负例。负例需要更谨慎构造（见 3.4）。
- 一个 `(subject, object)` 可能对应多个 `Pxxx`（多标签），关系预测建议采用**多标签**训练目标（sigmoid）。

### 3.4 负例（NA）与 hard negative（决定精度）

推荐策略：

- **类型约束负例**：若某关系要求 object 为地点，但 object 的类型（Wikidata `P31`/`P279`）明显不匹配，则该关系对该 bag 为强负例。
- **对比负例（hard negative）**：同一段落/同一 source 内，subject 与多个 object 共同出现时，用“类型相同但不在 KB 里成对”的 object 作为 hard negative（仍需谨慎，避免把缺失事实当负例）。

工程上建议把 NA 设计为：

- 只在“关系 allowlist 内全都不成立”的情况下给 NA（并记录理由：类型不匹配/无结构化支撑/证据不足等）。

---

## 4. 模型层：MIL-Attention 关系分类（限定域）

### 4.1 模型定位

MIL-Attention 在本项目中应当定位为：

- **一个限定域关系分类器**（输出属于哪些 `Pxxx`）
- 输入是 “一个 bag（同一 subject-object 对的多句证据）”
- 输出用于补充图中的边，与 `relation_patterns.py` 的规则抽取互补

### 4.2 编码器选择（追求效果：优先 BERT + entity markers）

若目标是“尽可能好”，推荐：

- **BERT/中文RoBERTa/多语模型** 作为句子编码器
- 在句子中插入 entity markers（如 `[E1]...[/E1] [E2]...[/E2]`）帮助模型聚焦实体对

中文优先的推荐实现：

- Encoder 优先选 **中文预训练模型**（例如 Chinese RoBERTa / MacBERT 一类），并在 DS 数据上微调；
- 对于中文句子里夹杂英文专名，不必切换多语模型：只要 EL 能把英文专名链接到正确 QID，模型仍然能学习到中文上下文触发的关系表达。

备选（算力受限时）：

- **PCNN（piecewise CNN）+ MIL（max/attention）**（经典 DS 方法，依赖更少、训练更轻量）

PCNN 的依据与定位：

- 经典工作：Zeng et al., *Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks*（EMNLP 2015，ACL Anthology：`https://aclanthology.org/D15-1203/`）
- 其核心是：用 **piecewise max pooling** 捕获两实体之间的结构片段特征，并在 **bag-level multi-instance learning** 下训练，缓解远程监督的 wrong-label 噪声。

在本项目中的推荐用法（“尽可能好”且难度合适）：

- **Phase 1**：PCNN + MIL-Attention 做强可复现 baseline（训练快、依赖轻、便于做 ablation）
- **Phase 2**：PCNN + attention（bag attention 或 relation-aware attention）作为中等成本提升
- **Phase 3**：若资源允许，再切到 BERT（通常效果上限更高），并与 PCNN baseline 做严格对比（同一 DS 数据与同一 gold）

### 4.3 Bag Attention（relation-aware）

对每个关系 `r` 学习一个 query 向量 `q_r`：

- 句子表示 `h_i = Encoder(x_i)`
- 注意力权重：`alpha_i^(r) = softmax(q_r^T h_i)`
- bag 表示：`H^(r) = sum_i alpha_i^(r) h_i`
- 关系得分：`p(r|bag) = sigmoid(W_r^T H^(r) + b_r)`

训练目标：

- 多标签 BCE（适配一个实体对多关系）
- 类别不均衡使用 `pos_weight` 或 focal loss

### 4.4 多中心训练策略（推荐两种之一）

路线 A（更稳、通常更高质量）：

- 按 `seed_type` 训练三个模型（Person/Concept/Award），每个仅覆盖其 allowlist 关系集合。

路线 B（更统一、工程复杂）：

- 一个共享 encoder，多头输出；对不允许关系做 mask（logits = -inf），并在训练时按 seed_type 选择 loss 计算范围。

第一阶段建议路线 A，便于隔离噪声并做对比实验。

---

## 5. 与现有规则抽取的融合（把系统做“强”）

### 5.1 融合原则：规则高精，MIL 高召回，最终以“可审计”保证质量

推荐融合策略：

- 如果规则抽取命中（`relation_patterns.py`），默认作为高置信边写入，`score_rule = 1.0` 或一个固定高值（并保留规则命中说明）。
- MIL 输出作为补充候选：
  - 设每个关系阈值 `tau_r`（在 gold dev 上调）
  - 通过类型约束过滤（若 object 类型冲突则丢弃或降权）
- 若同一 (s, r) 出现多个 object：
  - 支持多值属性（Wikidata 常见），但可增加去重与冲突策略（按 score、证据覆盖、来源优先级）

### 5.2 输出格式与 provenance

必须在写图与导出时明确区分来源：

- `provenance = "pattern_relation_extraction"`（已有）
- `provenance = "mil_relation_extraction"`

并保留：

- `snippet`（包含 top evidence + attention 权重摘要）
- `extraction_method`（如 `mil_attention_bert_person_v1`）

---

## 6. 评估与验收（没有 gold，就谈不上“尽可能好”）

### 6.1 必须建设的 gold 集（规模不必大，但要干净）

建议至少做两类 gold：

1) **句子级关系表达标注**（推荐）
- 采样：从 MIL 预测高分与规则命中样本中分层抽样
- 标注：给定句子 + 两实体（高亮/标记），判断是否表达某关系（以及是哪一个）
- 优点：可直接评估“证据句选择是否正确”，对 attention 的可解释性评估很关键

2) **bag 级关系标注**（可选）
- 对一个实体对的多句集合，标注其关系集合

### 6.2 指标（建议）

- 分类指标：micro/macro F1，per-relation Precision/Recall
- 证据指标：Top-1 evidence accuracy（attention 最高权重句是否为真证据）
- 工程指标：入图后人工抽检通过率（例如每个中心随机抽 50 条边，正确率≥某阈值）

---

## 7. 工程落地：模块、文件与接入点（建议实现清单）

### 7.1 新增/扩展的中间层文件（建议）

在 `data/curated/` 新增（或并入现有结构）：

- `bags.jsonl`：每行一个 bag（可训练/可审计）
- `ds_labels.jsonl`：每个 bag 的 DS 标签来源与证据（来自 Wikidata 的哪条声明）
- `re_predictions.jsonl`：MIL 推理输出（含 attention top-k）

### 7.2 建议新增模块（Python 包路径建议）

- `turing_kg/relation/schema.py`
  - 读取 `sources/relation_schema.json` 与 allowlist
  - 类型约束与关系 mask
- `turing_kg/relation/dataset.py`
  - 从 `sentences/routing/resolved` 生成 bags
  - 生成 DS 标签与 NA/hard negative
- `turing_kg/relation/model_mil_attention.py`
  - encoder + attention + multi-label heads
- `turing_kg/relation/train.py`
  - 训练入口（可配置：seed_type、关系子集、超参、输出路径）
- `turing_kg/relation/infer.py`
  - 推理入口：对一个 source 或全量跑，输出 `re_predictions.jsonl`
- `turing_kg/relation/ingest.py`
  - 把推理结果转成图边（GraphBuild）与 triples 行

### 7.3 接入点（不破坏现有流水线）

建议在 `turing_kg/build.py` 中新增一个可控开关：

- 现有：NER/EL + pattern RE
- 新增：在 `prels` 之后或并行，调用 `mil_infer` 产生 `mil_rels`，再 ingest

默认可设：

- 训练与推理分离（训练在独立脚本中完成）
- 构建图时只加载已训练模型并推理（避免 build 变得不可控）

### 7.4 去主体特化：把“图灵特化规则”改成“seed-aware 规则”（保证可移植）

本项目要可移植，规则抽取层必须做到“以 seed 配置驱动”，避免硬编码某个人名/某个实体。

建议对规则/模板 RE 做如下抽象（不要求一次性完成，但要作为后续改造目标写入路线）：

- **中文模板层**（正则槽位）：
  - 模板库按 `seed_type` 或 `prop_id` 组织，而不是按某个主体；
  - 模板触发可叠加 `seed_anchors_zh`（例如要求句子与 seed 有关联），但不要把具体人名写死。

- **英文依存层（若后续再启用英文）**：
  - 将当前“找 Turing 指称”的逻辑抽象为 `find_seed_head_token(doc, seed_anchors_en, seed_type)`；
  - 对不同 `seed_type` 采用不同 head 识别策略（Person 以 PERSON/专名为主；Concept 以名词短语；Award 以专有名词短语/ORG 组合）。

- **统一输出结构**：
  - 规则抽取输出必须与 MIL 输出同构（`subject_qid / prop_id / object_qid / sentence_idx / score / method / snippet`），以便融合与评估复用。

---

## 8. 实现里程碑（建议按“质量优先”的顺序）

Milestone 0（Schema 与配置）：

- 完成 `relation_schema.json` 与 allowlist（先覆盖 10–30 个核心关系）
- 为三类 seed_type 给出初版允许关系集合与类型约束

Milestone 1（bag 构造 + DS 标签 + 审计落盘）：

- 生成 `bags.jsonl` + `ds_labels.jsonl`
- 抽样检查：每个中心 30 个 bag，人工确认 bag 内容与标签合理

Milestone 2（PCNN + MIL-Attention baseline）：

- 用同一个 encoder（可以先用轻量句向量）实现 bag max pooling
- 作为可复现 baseline，建立训练/评估闭环

Milestone 3（MIL-Attention 正式版）：

- BERT + entity markers + relation-aware attention
- 做阈值、类型过滤、与规则抽取融合

Milestone 4（Gold 集与系统级验收）：

- 建立 gold dev/test
- 出 per-relation 指标 + evidence 指标
- 输出入图质量抽检报告

---

## 10. 迁移到“其他关键主体”的操作清单（把移植成本降到最低）

本节是为了确保你后续把系统移植到新的主体（例如换成其他人物/理论/奖项/机构）时，工作量可控且可复现。

### 10.1 只改配置（理想目标）

当以下条件满足时，你应当只需要修改配置即可迁移：

- `sources/seed_entities.json`：新增/替换 seeds（QID、中文 anchors、触发词）
- `sources/relation_allowlist.json`：按 seed_type 调整关系子集
- `sources/relation_schema.json`：必要时补充关系定义与类型约束
- `sources/attribution_config.json`：路由阈值/窗口参数（控制抽取门控与噪声）

### 10.2 需要少量补模板/规则（常见、可接受）

以下属于“正常的迁移成本”，但仍保持工程可控：

- 为新主体领域补充少量中文模板（例如“提出/发明/首次提出/定义为/属于/用于”等领域句式）
- 为某些关系补充更强的负例过滤（类型/触发词）

### 10.3 什么时候必须改代码（应尽量避免）

如果出现以下情况，说明系统仍存在主体特化，应当优先把特化抽象成配置：

- 抽取器/模型中出现主体名称硬编码（例如 `Turing`、固定 QID）
- 依存/规则层的 anchor/head 识别逻辑只能适配某一类 seed
- 输出中间层缺关键字段导致无法跨主体复跑对比（缺 `seed_id/subject_qid` 等）

---

## 11. 实现规格（Implementation Spec，中文主干 + PCNN 第一阶段）

本节把“建议”落到**可直接实现**的工程规格，避免后续编码时再次做设计决策。

### 11.1 语言筛选（中文主干阈值固定为 0.3）

- **中文句子判定**：句子中文字符占比 \(\ge 0.3\) 视为中文句子。
- **策略**：
  - 训练/推理仅使用中文句子；
  - 允许句子中夹杂英文专名（只要 EL 能归一到同一 QID）。

实现建议：

- 复用 `relation_patterns.py` 中的 `_zh_ratio` 思路（或在 `turing_kg/relation/text_lang.py` 新建同名函数），阈值写入配置（但默认 0.3）。

### 11.2 稳定键（所有 RE 中间层必须可 join）

所有 RE 中间文件必须包含以下稳定键字段（最少集合）：

- `source_id`：稳定文本源 ID（与 `sentences.jsonl` 一致）
- `source_url`：可空，但尽量保留
- `source_label`：可空，但尽量保留（便于审计）
- `citation_key`：可空（但若已有 bibliography，应保留）
- `seed_id`：来自 `seed_entities.json`
- `seed_qid` / `subject_qid`：两者至少一个，推荐同时保留（`subject_qid == seed_qid`）
- `object_qid`：目标实体 QID（必须是 `Q...`，否则作为 literal 处理，不进入 MIL）
- `sentence_idx`：在该 source 内的句子索引

推荐派生键：

- `bag_id = f"{source_id}|{seed_id}|{object_qid}"`（同一 source 内，同一 seed 与 object 的 bag）

### 11.3 中间层文件协议（JSONL）

本项目遵循“**一行一个对象**”的 JSONL 约定，字段尽量扁平、可 grep、可增量写。

#### 11.3.1 `data/curated/bags.jsonl`

用途：MIL 训练与推理的最小输入（bag 级）。

每行一个 bag（同一 `source_id + seed_id + object_qid`）。

字段：

- `bag_id`（str，必需）
- `source_id`（str，必需）
- `source_url`（str，可空）
- `source_label`（str，可空）
- `citation_key`（str，可空）
- `seed_id`（str，必需）
- `subject_qid`（str，必需）
- `seed_type`（str，必需，来自 seed 配置的 `type`）
- `object_qid`（str，必需，必须 `Q...`）
- `object_mention_set`（list[str]，可选；bag 内不同 mention 的去重集合，便于审计）
- `instances`（list[object]，必需；bag 内句子实例）
  - `sentence_idx`（int，必需）
  - `sentence`（str，必需）
  - `routing`（object，可选；来自 `routing.jsonl` 的 top-1 或 top-k 归因）
    - `score`（float）
    - `reasons`（dict）
  - `mentions`（object，可选；用于记录这句中 object 的 mention 与 NER）
    - `object_mention`（str）
    - `ner_label`（str，可空）

示例（仅示意）：

```json
{"bag_id":"file:xxx|turing_person|Q127992","source_id":"file:xxx","source_url":"","source_label":"zh:某文章","citation_key":"CIT001","seed_id":"turing_person","subject_qid":"Q7251","seed_type":"Person","object_qid":"Q127992","object_mention_set":["ACM","Association for Computing Machinery"],"instances":[{"sentence_idx":12,"sentence":"……图灵奖由美国计算机协会（ACM）颁发……","routing":{"score":0.78,"reasons":{"anchor_hits":["图灵奖"]}},"mentions":{"object_mention":"ACM","ner_label":"ORG"}}]}
```

#### 11.3.2 `data/curated/ds_labels.jsonl`

用途：为每个 bag 生成远程监督（DS）弱标签，并保留来源证据。

每行一个 bag 的标签对象（与 `bags.jsonl` 一一对应），字段：

- `bag_id`（str，必需）
- `seed_id` / `subject_qid` / `object_qid` / `seed_type`（同上，必需）
- `labels_pos`（list[str]，必需；正关系 `prop_id` 列表，如 `["P166","P108"]`）
- `labels_space`（list[str]，必需；该 `seed_type` 的 allowlist prop_id（用于 mask））
- `label_source`（object，必需）
  - `kb`（str，必需，固定 `"wikidata"`）
  - `matched_triples`（list[object]，可选；每个 prop_id 的匹配来源）
    - `prop_id`
    - `evidence`（简短说明，如“structured layer claim matched”）
- `negatives`（object，可选；用于训练时的负例信息与审计）
  - `type_incompatible_props`（list[str]）
  - `hard_negative_note`（str，可空）

示例：

```json
{"bag_id":"file:xxx|turing_person|Q127992","seed_id":"turing_person","seed_type":"Person","subject_qid":"Q7251","object_qid":"Q127992","labels_pos":["P166"],"labels_space":["P19","P20","P69","P108","P1416","P166","P27","P106","P551","P800"],"label_source":{"kb":"wikidata","matched_triples":[{"prop_id":"P166","evidence":"wikidata (Q7251,P166,Q127992)"}]},"negatives":{"type_incompatible_props":["P19","P20"]}}
```

#### 11.3.3 `data/curated/re_predictions.jsonl`

用途：推理输出与入图前审计（bag 级预测 + 证据句）。

每行一个 bag 的预测对象：

- `bag_id`（str，必需）
- `seed_id` / `subject_qid` / `object_qid` / `seed_type`（必需）
- `model`（str，必需；例如 `pcnn_milatt_Person`）
- `predictions`（list[object]，必需）
  - `prop_id`（str）
  - `score`（float，0~1）
  - `passed_threshold`（bool）
  - `threshold`（float）
  - `evidence`（object，必需）
    - `top_sentence_idx`（int）
    - `top_sentence`（str）
    - `top_sentence_score`（float，MIL-Attention 下为注意力在 `attention_instance_idx` 上的权重）
    - `topk`（list[object]，可选；attention 版本保留 top-k）

示例：

```json
{"bag_id":"file:xxx|turing_person|Q127992","seed_id":"turing_person","seed_type":"Person","subject_qid":"Q7251","object_qid":"Q127992","model":"pcnn_milatt_Person","predictions":[{"prop_id":"P166","score":0.91,"threshold":0.75,"passed_threshold":true,"evidence":{"top_sentence_idx":12,"top_sentence":"……图灵奖由美国计算机协会（ACM）颁发……","top_sentence_score":0.42,"attention_instance_idx":0}}]}
```

### 11.4 关系配置加载与 mask（工程必做）

配置文件：

- `sources/relation_schema.json`
- `sources/relation_allowlist.json`

运行时规则：

- 读取 `seed_type` 对应的 allowlist `labels_space`
- **训练与推理只在 `labels_space` 内计算 loss / 输出**（其余 prop_id 直接 mask 掉）

伪代码（逻辑规范）：

```text
space = allowlist[seed_type]  # list[prop_id]
pos = ds_labels.labels_pos ∩ space

logits = model(bag)  # dict[prop_id -> logit] 或 tensor over space
logits_masked = logits restricted to space
loss = BCEWithLogits(logits_masked, y over space)  # multi-label
```

### 11.5 PCNN + MIL-Attention（第一阶段 baseline）实现要点（中文）

目标：在“难度合适”的前提下，给出一个**强可复现**的 DS 关系抽取模型，并为后续 attention/BERT 做对比基线。

#### 11.5.1 输入表示（推荐：字粒度或词粒度二选一）

中文工程实践建议优先考虑 **字粒度**（避免分词误差扩散），但若你已有稳定分词器也可用词粒度。

每个句子 instance 需要：

- `tokens`: list[str]（字或词）
- `pos1`: 相对 subject anchor 的位置嵌入索引（可选；若暂时无法精确定位 subject span，可退化为“句首/句中”或仅使用 object 位置）
- `pos2`: 相对 object mention 的位置嵌入索引（推荐；object span 更容易由 EL mention 定位）
- `mask_segments`: 三段分割边界（piecewise pooling 需要）

实践折中（符合当前项目数据）：

- 第一阶段可先做 **仅定位 object mention** 的 PCNN（subject 以 seed 归因保证主题相关），piecewise 以 object 为中心简化；
- 若后续补齐 char span（推荐），再升级为标准双实体 piecewise 分段。

#### 11.5.2 网络结构（最小可复现）

- `emb(token)`：随机初始化或用预训练字向量（可选）
- `emb(pos)`：位置嵌入（pos1/pos2）
- 卷积：多窗口（例如 3/4/5）+ ReLU
- **piecewise max pooling**：按实体位置切成 3 段，分别 max，拼接成句子向量 `h_i`
- 句子级分类：对每个关系输出 logit（或共享 `W`）
- **MIL-Attention（Lin et al. 风格，共享 u）**：对 bag 内句向量 `h_i` 计算 `alpha = softmax(u^T tanh(W h_i))`，`h_bag = sum_i alpha_i h_i`，再经共享 `fc` 得 bag 级 logits（**不再使用 MIL-max**）

默认超参建议（可写入 `sources/relation_model_config.json`，若暂不建文件可先硬编码在训练脚本）：

- `max_sentence_len`: 128（中文句子通常够用）
- `embedding_dim`: 100~200
- `pos_emb_dim`: 5~20
- `conv_filters`: 230（经典设定），或 128（更轻）
- `dropout`: 0.5
- `optimizer`: Adam
- `lr`: 1e-3（PCNN）/ 若训练不稳可降 3e-4
- `batch_size_bags`: 16~64（看内存）
- `max_bag_size`: 10（bag 内采样，避免超长 bag）

#### 11.5.3 训练采样（保证工程稳定）

- **bag 采样**：每个 bag 最多取 `max_bag_size` 句，训练时随机采样，推理时可全量或同样采样并取最大
- **类别不均衡**：对每个 prop_id 计算 `pos_weight`，使用 BCEWithLogits 的 `pos_weight` 或 focal loss
- **NA 处理**：
  - 第一阶段建议不显式建一个 “NA 类” softmax；
  - 直接做 multi-label sigmoid，仅对 allowlist 关系输出分数，推理时用阈值决定是否入图。

### 11.6 阈值校准与落盘（入图质量的关键）

阈值文件建议新增：

- `sources/relation_thresholds.json`

结构：

- `by_seed_type`：每个 seed_type 各自一份 `prop_id -> threshold`
- `default_threshold`：缺省阈值（例如 0.8）

阈值来源（工程可执行）：

- 在 gold dev 上按 prop_id 做 PR 曲线，选使 F1 最大或 precision≥目标值的阈值。

### 11.7 推理入图的规范（统一与可审计）

入图前过滤顺序（建议）：

1) `prop_id` 必须在 allowlist（mask）
2) `score >= threshold[prop_id]`
3) 类型约束过滤（若能拿到 object 的 `P31/P279`，否则先用 NER 粗类型）
4) 去重：同一 `source_id + seed_id + prop_id + object_qid` 去重

写入图边字段建议：

- `provenance="mil_relation_extraction"`
- `prop_id / prop_label`（来自 schema）
- `snippet`：包含 top evidence sentence 与模型分数
- `citation_key / source_url`：保留溯源

---

## 9. 需要你确认/决策的事项（请逐项明确，否则实现会偏离目标）

1) **关系范围**：第一阶段你希望覆盖哪些 Wikidata 属性（`Pxxx`）？
   - 建议你给出一个“必须覆盖的 Top 10–30”清单（Person/Concept/Award 分别列）

2) **语言范围**：MIL 模型是要：
   - 仅中文（推荐第一阶段）？仅英文？还是中英混合（multilingual encoder）？

3) **bag 定义范围**：你希望 bag 的证据句来源是：
   - 仅同句共现（高精）？
   - 允许窗口共现（高召回，但需更强过滤）？

4) **训练资源与依赖**：
   - 是否允许引入 `transformers` / `torch` 等深度学习依赖？
   - 可用显存/训练时间大概是多少（决定 BERT/批大小/蒸馏策略）？

5) **负例策略的“保守程度”**：
   - 是否接受把“KB 缺失”视为弱负例（会带来 label noise）？
   - 还是只用类型约束+hard negative（更保守，但训练数据更少）？

6) **评估目标**：
   - 你更看重：入图精度（precision）还是覆盖（recall）？
   - 是否必须对每条边输出可解释证据（Top-k 句子）作为验收硬指标？

7) **多中心训练策略**：
   - 你倾向三个独立模型（更稳）还是一个共享模型（更统一）？

8) **中文主干的语料筛选规则**（为保证 RE 质量，建议明确）：
   - “中文为主”如何判定（例如按句子中文字符占比阈值）？
   - 对混杂英文专名的中文句子是否保留（建议保留）？

---

## 附：推荐的“第一版关系 allowlist”方向（示例，不是最终结论）

说明：下面仅用于启发，最终以你确认的 Pxxx 清单为准。

下面给出一个**更完整的、分阶段**的建议（请按你的实验资源与报告篇幅裁剪）。

### A. Phase A（强基线，10–20 条/seed_type）

- Person（图灵）建议至少覆盖：
  - **P19** 出生地、**P20** 逝世地、**P69** 毕业院校、**P108** 雇主、**P1416** 隶属机构、**P166** 所获奖项、**P27** 国籍、**P106** 职业、**P551** 居住地
  - （可选）**P800** 代表作（作品）、**P737** influenced by（受影响于，较噪声，后置）

- Concept（图灵机）建议至少覆盖：
  - **P279** 上位概念（subclass of）、**P361** 所属整体（part of）
  - （可选）**P31** 实例（instance of，用作约束也可）

- Award（图灵奖）建议至少覆盖：
  - **P31** 实例、**P361** 所属整体/系列、**P127** 拥有者（owner）、**P137** 运营方（operator）、**P17** 国家、**P276** 地点（若有）

### B. Phase B（扩覆盖，额外 10–30 条/seed_type，需更强过滤）

Person（图灵）可扩展方向（从 Wikidata Q7251 的“常见关系语义”出发）：

- **学术/机构链路**：P69（已含）、P108（已含）、P1416（已含）、P463 member of（会员/学会）、P512 学位（若要文字抽取需谨慎）
- **作品/产出**：P800 代表作、P2860 cites（引用，通常不适合文本抽取第一阶段）、P50 作者（需要从作品侧建模）
- **家庭/人物关系（噪声高，建议后置）**：P26 配偶、P40 子女、P3373 siblings

Concept（图灵机）可扩展方向（更偏本体/百科事实）：

- **P2579** studied by（被研究者：人/组织）、**P366** use（用途/应用）、**P1269** facet of（方面/分支）
- 以及与你后续“学科本体”对齐的关系（若接 CSO/DBpedia/领域本体，可作为候选映射）

Award（图灵奖）可扩展方向（从 Wikipedia 结构化字段画像出发）：

- **“Presented by / sponsor / reward / first award / awarded for”** 这些信息在自然语言中频繁出现，但在 Wikidata 中对应属性可能不同且有时为字面值或限定值；建议先把它们作为：
  - 规则抽取或结构化补全（更稳）
  - 或作为 MIL 的辅助特征/过滤条件（而不是一开始就纳入大量新 `Pxxx`）

### C. 重要提醒：奖项的“获奖者列表”不建议第一阶段从文本直接抽

原因：

- Wikidata 中奖项与获奖者的建模经常是“人侧 P166 指向奖项”，而不是“奖项侧一个稳定属性指向所有获奖者”。
- 因此更稳的实现是：优先抽取/对齐 **Person 的 P166**，再在图查询层面反向得到“奖项有哪些获奖者”。

---

## 12. 实现进度与后续步骤（代码落地）

### 12.1 已完成

| 步骤 | 产物 | 说明 |
|------|------|------|
| 1 | `data/curated/bags.jsonl` | `turing_kg/relation/bags.py`，`scripts/build_relation_bags.py`；中文门控默认 0.3 |
| 2 | `data/curated/ds_labels.jsonl` | `turing_kg/relation/ds_labels.py`，`scripts/build_ds_labels.py`；`labels_space` 来自 `sources/relation_allowlist.json`，`labels_pos` 来自 Wikidata subject 声明与 object 对齐 |
| 3 | `sources/relation_thresholds.json` | 默认阈值 + 按 `prop_id` 覆盖（初稿已入库） |
| 4 | PCNN + MIL-Attention 训练 | `turing_kg/relation/pcnn_mil.py`（`PCNNMILAttention`）、`pcnn_train.py`；`scripts/train_relation_pcnn.py`（可选 `--att-dim`）；**需在 `.venv` 中安装 `torch`**；旧版 MIL-max checkpoint 需**重新训练** |

运行顺序（在有一次完整 `run.py` 产出 `resolved.jsonl` 之后）：

1. `python scripts/build_relation_bags.py`
2. `python scripts/build_ds_labels.py`（需联网访问 Wikidata API，已走 `wikidata_api` 缓存）
3. **安装依赖（若尚未）**：`.venv\Scripts\pip.exe install -r requirements.txt`
4. **训练（示例）**：`.venv\Scripts\python.exe scripts\train_relation_pcnn.py --seed-type Person`  
   或 PowerShell：`.\scripts\train_relation_pcnn.ps1 --seed-type Person`

### 12.2 待实现（建议顺序）

| 步骤 | 内容 |
|------|------|
| 5 | **推理与导出**（已实现） | `turing_kg/relation/pcnn_infer.py`：`infer_and_write` / `infer_all_available`；`scripts/infer_relation_pcnn.py`；单类型 → `re_predictions_pcnn_{Person\|Concept\|Award}.jsonl`，省略 `--seed-type` → 合并 `re_predictions.jsonl` |
| 5b | **三元组 CSV**（已实现） | `turing_kg/relation/mil_ingest.py`：`export_mil_triples_from_file` → `data/triples_mil_extracted.csv`；`scripts/export_mil_triples.py` |
| 6 | **与主构建融合**（已实现） | `export_all`（`turing_kg/build.py`）调用 `apply_mil_to_export_if_present`：自动合并 `triples_mil_extracted.csv` 到 `triples_extracted.csv`（经 `merge_triple_rows`）；并将 `re_predictions*.jsonl` 中过阈值的边写入 `GraphBuild`（与同 `(subject,object,prop_id)` 去重） |
| 7 | **Gold 集与阈值校准**：按 dev 调 `relation_thresholds.json` |

### 12.3 第二步实现说明（DS 标签）

- 对每个 bag，在 **subject 实体的全部 item 型声明** 中查找 `object_qid`；若 `prop_id` 落在该 `seed_type` 的 allowlist 内，则记入 `labels_pos`。
- `labels_pos` 为空表示该实体对在 KB 的 allowlist 下无结构化正例，**仍保留该行**，供多标签训练中的负信号或分析；具体训练时是否做重采样由 `train_pcnn` 配置决定。

### 12.4 文档目标 vs 当前代码：哪些算「做完」、哪些还是路线图

下列区分 **「工程闭环已具备」**（可跑通、可落盘、可并入 `run.py`）与 **「文档中的理想形态 / 尚未实现」**。

| 文档中的能力 | 当前状态 |
|-------------|----------|
| 统一 schema + allowlist（`relation_schema.json` / `relation_allowlist.json`） | **已实现**（配置文件已入库，DS/MIL 训练用 allowlist） |
| bags / ds_labels / re_predictions 中间层 | **已实现** |
| 远程监督 + **PCNN + MIL-Attention**（共享 selective attention，替代 MIL-max） | **已实现**（`PCNNMILAttention`、`forward_bag`；checkpoint 含 `aggregation: mil_attention`） |
| 每关系一条 **relation-specific attention**（Lin 全文逐关系 u） | **未实现**；当前为**共享** bag 注意力（对所有关系共用一个 `h_bag`），与常见单阶段多标签实现一致 |
| **BERT** 编码器 + 实体标记 | **未实现**（路线图，需 `transformers` 与更多算力） |
| 推理阶段 **类型约束**（object 的 P31/NER 与关系 schema 对齐过滤） | **未实现**；导出仅按 `score >= threshold` |
| **`relation_model_config.json`** 外置超参 | **未实现**；超参写在 `pcnn_train.py` 默认值内 |
| **`relation_patterns.py` 英文侧完全 seed-aware、去主体特化** | **未改**；仍存在 Turing 锚点特化，移植其他主体前需重构（见 §0.1 / §7.4） |
| **Gold dev/test + 阈值校准流程**（§6、§11.6） | **未自动化**；需人工标注 + 手工调 `relation_thresholds.json` |
| OpenIE / Phase C | **未做**（文档定位为后置） |

**结论**：若以「限定域 RE + DS + PCNN + MIL-Attention + 导出 + 并入主构建」为范围，**主链路已闭合**；BERT、gold 自动校准、规则层完全可移植等仍见上表。

### 12.5 为何大多数 MIL 预测「没过阈值」——是代码写错了吗？

**不是阈值判断逻辑写错**：`export_mil_triples` 与 `pcnn_infer` 中 `passed_threshold = (score >= threshold[prop_id])` 与配置文件一致，行为符合设计。

数量少（例如 43 个 bag、最终只导出个位数三元组）通常来自 **设定与模型输出的错配**，而非单点 bug：

1. **阈值量级**：`relation_thresholds.json` 中 **0.72～0.78** 面向「高置信、可入库」；当前 **字级 PCNN、随机初始化 embedding、小样本训练** 下，**sigmoid 分数整体偏低**（常见在 0～0.1）是正常现象，与 0.75 线不匹配时，通过率会极低。
2. **任务难度**：多数 `(subject, object)` bag 在 KB 中 **本无对应关系**（`labels_pos` 常为空），模型应打 **低分**；若阈值仍按「正例精度」设，负例会几乎全部滤掉，正例也可能因未校准而上不去。
3. **训练目标**：多标签 BCE + `pos_weight` 在 **bag 极少** 时易 **过拟合 DS**，对未见组合泛化差，推理分数尺度不稳定。
4. **文档已写但未落地的环节**：§11.6 要求 **在 gold dev 上按 PR 曲线调阈值**——未做时，沿用默认 0.75 必然与真实分数分布脱节。

**改进方向（按优先级）**：在 dev 上统计分数分布并 **下调阈值** 或 **per-prop 校准**；增加训练数据；换 **预训练字向量 / BERT**；可选升级为 **逐关系 attention**；补 **gold 集** 再固定阈值。
