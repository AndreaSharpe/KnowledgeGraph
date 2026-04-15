## 目标

把当前项目的“实体链接（EL）”工程化升级为一条可审计、可复现、可检查的流水线，并补齐你现在最缺的两类能力：

- **标准化中间数据层**：让清洗/分段/分句/路由/抽取/链接每一步都能落盘检查与复跑对比
- **局部 + 全局（协同）实体链接**：在已有候选检索（Wikidata Search + `entity_map`）基础上，增强局部打分，并在文档窗口内做协同一致性解码

同时为后续“加语义向量检索/重排”和“CRF 标注与训练验收”留出稳定的扩展点。

---

## 设计原则（工程化约束）

- **分层清晰**：raw → processed → curated → exports
- **可审计**：每个 QID 的选择必须能解释（local 分数、global coherence 贡献、先验命中等）
- **可复现**：所有参数来自配置文件；每一步都可单独运行并产生确定的输出文件
- **低侵入**：不推倒现有 `turing_kg/build.py`，以“新增模块 + 接入点”的方式落地

---

## 决策记录：句子门控抽取 vs 文档级全覆盖抽取

本项目当前采用 **“句子路由/归因 →（可选）邻接窗口扩展 → 对被选句子做 NER/EL/RE”** 的门控策略，其动机是：

- **降噪**：避免对整篇文本全句抽取导致大量无关实体进入图谱
- **可归因**：每条抽取结果能确定性绑定到某个 `sentence_idx` 与 seed 归因（审计更直接）

但它的代价是：像“他就职于/他在…任职”这类句子如果 **不含 anchor/trigger 且离 anchor 较远**，可能仅靠窗口覆盖而漏掉。

未来若要更“经典的文档级抽取”，可改为：

- **对每个 source 的所有句子全覆盖跑 NER（与可选 EL）**
- 再在后处理阶段用 `routing.jsonl` 的路由分数把 mention/关系 **归因到 seed**（top-1 或 top-k）
- 用阈值与实体-主题一致性（类型/图邻域 coherence）做过滤，抵消全覆盖带来的噪声

该改造会提升召回，但必须同步强化：中间层落盘、过滤策略、以及 reasons 审计（local + global）。

## 一、标准化中间数据层（强优先级）

### 1.1 新增目录与落盘约定

建议新增：

- `data/processed/`：清洗/切分/路由后的“句子级”数据
- `data/curated/`：mention、候选、最终链接、协同解码后的“链接级”数据

所有中间文件尽量用 **JSONL**（每行一个对象），利于增量写入与 grep/抽查。

### 1.2 统一的键（最重要）

为保证各步可 join、可对照，定义最小键集合：

- `source_id`：文本源的稳定 ID（来自 `TextSource.source_id` 或等价字段）  
- `paragraph_idx`：段落索引（没有段落时可设为 0）
- `sentence_idx`：句子索引（在该 source 内从 0 递增）
- `sentence`：句子文本（清洗后的）
- `char_span`：可选，mention 在句子中的字符起止（便于回看）

### 1.3 建议新增的中间文件（最小可行）

1. `data/processed/sentences.jsonl`

- 目的：检查清洗与分句质量
- 字段建议：
  - `source_id`, `paragraph_idx`, `sentence_idx`, `sentence`
  - `lang`（可选）
  - `meta`（可选：页码/章节等）

1. `data/processed/routing.jsonl`

- 目的：检查句子路由（归因）效果
- 字段建议：
  - `source_id`, `sentence_idx`
  - `seed_id`, `seed_qid`
  - `score`, `reasons`（结构化 dict，含 anchor/trigger/window 等）
  - `selected`（是否进入 top-k / 是否被采样进抽取）

1. `data/curated/mentions.jsonl`

- 目的：检查 NER 输出与 mention 规范化
- 字段建议：
  - `source_id`, `sentence_idx`
  - `mention`, `ner_label`, `char_start`, `char_end`
  - `seed_id`（若是多归因抽取，记录该条 mention 属于哪个 seed 子任务）
  - `ner_backend`（spacy/crf）

1. `data/curated/candidates.jsonl`

- 目的：检查候选生成与初排是否合理（EL 的“召回与先验”）
- 字段建议：
  - `source_id`, `sentence_idx`, `mention`
  - `candidates`: `[{"qid": "...", "label": "...", "description": "...", "aliases": [...], "retrieval": {...}, "prior": {...}}]`
  - `entity_map_override`（若命中，写明命中理由）

1. `data/curated/resolved.jsonl`

- 目的：检查最终链接选择，以及 local/global 分数贡献
- 字段建议：
  - `source_id`, `sentence_idx`, `mention`
  - `chosen_qid`
  - `scores`: `{ "local": 0.73, "global": 0.12, "total": 0.85 }`
  - `reasons`: `{ "local": {...}, "global": {...}, "prior": {...} }`

验收方式：随机抽 20 条 `resolved.jsonl`，能从 `sentences.jsonl` 找到原句，从 `candidates.jsonl` 看到候选列表与分数，从 `routing.jsonl` 看到它为什么归到某个 seed。

---

## 二、强化局部实体链接（统计型优先）

你当前的“Wikidata Search + 简单重排”可以保留为候选生成与初排；局部链接增强建议先做“统计型打分”，工程成本低且可解释。

### 2.1 局部打分（Local Scoring）建议配方

对每个 mention 与候选实体 e，构造候选文本：

- `entity_text(e) = label + aliases + description`（用空格拼接）

对 mention 侧构造上下文文本：

- `mention_text = mention + sentence (+ neighbor sentences 可选)`

局部打分可先用线性组合：

- **char‑ngram 相似度**：对 `mention` 与 `label/aliases` 做 3-5gram Jaccard/TF-IDF cosine
- **BM25**：把 `entity_text(e)` 当成“文档”，用 `mention_text` 当 query 做 BM25 打分（或反过来做重排）
- **类型兼容**：`ner_label` 与候选实体 instance-of/类型的软匹配（拿不到类型时先跳过）

输出：

- `local_score`
- `local_reasons`（命中 alias、ngram 覆盖率、BM25 主要命中词等）

#### 当前实现落地点（2026-04）

项目现已在 `turing_kg/linking/entity_linking.py` 的候选重排中加入 **char‑ngram Jaccard（n=3）** 作为统计型局部信号，并通过 `link_mention_with_candidates(...).local_breakdown` 将以下分解指标写入 `data/curated/resolved.jsonl`：

- `label_exact`
- `char_ngram_jaccard_3`
- `context_jaccard`

### 2.2 关于“先用统计、再加语义向量”

建议工程上把 local 分两种后端，配置选择：

- `local_ranker = "stat"`（默认）
- `local_ranker = "hybrid"`（统计 + 语义）

这样你可以在报告里明确对比“stat vs hybrid”。

---

## 三、加入语义向量检索/重排（可选增强）

你写的思路：

> Candidates = BM25(top 50) ∪ Embedding ANN(top 50)

### 3.1 结论：这个组合方式是对的（但要明确它在哪一层发生）

在“有本地实体语料库 + 检索索引”的前提下，常见做法就是 **混合检索（hybrid retrieval）**：

- BM25 擅长字面匹配
- ANN（向量近邻）擅长语义相似
- 并集能提升 recall，再用 local reranker 做最终排序

### 3.2 工程前提（你需要先具备）

要做 BM25/ANN 候选并集，你需要一个“实体候选语料库”，例如：

- 从 Wikidata 预先拉取一个实体子集（与你的主题相关的 QID 集合）
- 为每个实体存储 `label/aliases/description`，形成一个可索引 corpus

否则如果你完全依赖“在线 Wikidata search API”，那 BM25/ANN 很难在“全库”意义上做检索，只能做“对 API 返回候选的重排”。

### 3.3 推荐落地策略（先易后难）

先做：

- **对 API 返回的 topK 候选做统计重排**（char-ngram/BM25 on candidate texts）

再做（加分项）：

- 维护本地 `entity_corpus.jsonl`（仅收录你需要的实体范围）
- 做 BM25 index + ANN index
- 用并集得到候选，再进入 local reranker

---

## 四、协同/全局实体链接（Collective EL）

### 4.1 “全局”是什么范围

推荐先定义一个可控窗口：

- 同一 `source_id` 的同一段落
- 或者 sliding window：连续 N 句

### 4.2 全局一致性（coherence）怎么计算（可解释优先）

对两个候选实体 e_i, e_j 计算 coh(e_i,e_j)，可先做轻量特征：

- **共享属性/关系**：是否共享某些 Wikidata 属性值（如同组织、同奖项、同领域）
- **邻域重叠**：一跳邻居集合的 Jaccard（需要缓存邻居）
- **类型兼容**：同一语境中 PER/ORG/LOC 的组合是否合理（弱特征）

### 4.3 解码（先用贪心迭代，足够工程化）

输入：一个窗口内 mentions，每个 mention 有 topK 候选与 `local_score`

目标：

- 最大化 `sum(local_score) + lambda * sum(pairwise_coherence)`

实现（MVP）：

- 初始化：每个 mention 选 local_score 最大的候选
- 迭代：逐个 mention 尝试替换为次优候选，只要全局目标上升就接受，直到收敛/迭代上限

输出审计：

- 哪些 mention 被协同改变了选择（before/after）
- global 贡献是多少（便于写报告）

#### 当前实现落地点（2026-04）

已新增：

- 配置：`sources/entity_linking_config.json`（`collective.enabled/window_sentences/top_k_candidates/lambda_coherence/coherence_props`）
- 协同解码：`turing_kg/linking/collective_linking.py`
  - coherence：基于 Wikidata `wbgetentities` 的 claims，一跳邻域按 `coherence_props` 过滤后做 Jaccard
  - 解码：窗口内贪心坐标上升（coordinate ascent，最多 3 轮）
- 接入：`turing_kg/build.py` 在 `ingest_linked_spans(...)` 前对 `spans` 做协同回写
- 审计：协同信号写入 `data/curated/resolved.jsonl` 的 `reasons.global`，并在发生变化时记录 `changed_from/changed_to`

---

## 五、标注数据（CRF NER）与验收闭环

这块的核心是“数据与模型产物”，不是再写更多代码：

- `data/ner_crf/*.conll`（至少 train/dev）
- `models/crf_ner.pkl`
- 跑一次指定 PDF 1–50 页，落盘 `mentions.jsonl` 与最终 `triples_extracted.csv` 对比（spacy vs crf）

---

## 六、建议的实现顺序（可迭代交付）

### Phase 0：Schema 与落盘（1 次提交就能看到收益）

- 实现 `sentences.jsonl` 与 `routing.jsonl`
- 让你能检查清洗/分句/归因

### Phase 1：EL 可审计化（把“链接”变成工程模块）

- 落盘 `mentions.jsonl`、`candidates.jsonl`、`resolved.jsonl`
- reasons 结构化

### Phase 2：局部链接增强（统计型）

- char‑ngram +（可选）BM25 重排
- 输出稳定可解释的 local_score

### Phase 3：协同链接（collective）

- coherence 特征 + 贪心解码
- reasons 记录 global 贡献与 before/after

### Phase 4：语义向量候选（可选加分）

- 建本地实体语料库 + ANN 索引
- Candidates = BM25(top50) ∪ ANN(top50)，再 local rerank

---

## 附：运行解释器一致性（Windows/.venv）

本项目在 Windows 下建议统一使用 `.venv\Scripts\python.exe` 运行，避免出现“你安装了依赖但运行时却走系统 Python”的不一致问题。

已提供脚本：

- `scripts/run_venv.ps1`：使用项目根目录 `.venv` 执行 `run.py`

## 七、你最终可以怎么写“项目能力亮点”（面向答辩/展示）

- **工程化流水线**：raw→processed→curated→exports，中间层可审计可复现
- **EL 分层清晰**：candidate generation → prior ranking → local disambiguation → collective coherence
- **可解释**：每个链接决策的 reasons（local + global）
- **可对比实验**：stat local vs hybrid local；local-only vs collective

