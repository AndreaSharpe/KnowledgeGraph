# 图灵知识图谱

> **知识工程课程个人项目**  
> 以图灵为核心，构建**多中心种子（Person / Concept / Award）知识图谱。**

---

## 二、项目结构

```bash
KnowledgeGraph/
├── README.md                       # 项目文档
├── run.py                          # 项目入口
├── requirements.txt
├── docs/                           # 设计文档与旧版README
│   └── README.md                   # 旧版说明文档
│   └── IMPLEMENTATION_*.md
├── sources/                        # 配置中心
│   ├── seed_entities.json          # 多中心种子定义（anchors、triggers、type）
│   ├── web_sources.json
│   ├── pdf_sources.json
│   ├── bibliography.json
│   └── *config*.json
├── turing_kg/                      # 核心包（当前最核心实现）
│   ├── build.py                    # 核心构建流水线（端到端主逻辑）
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
│   ├── curated/                    # 可审计中间层
│   │   ├── mentions.jsonl
│   │   ├── resolved.jsonl
│   │   ├── re_predictions_pcnn_*.jsonl
│   │   ├── bags.jsonl
│   │   └── ds_labels.jsonl
│   ├── processed/                  # 预处理中间层
│   │   ├── sentences.jsonl         # 句子级样本（全源共享）
│   │   └── routing.jsonl           # 多中心归因记录
│   ├── nodes.csv
│   ├── relationships.csv
│   ├── triples_extracted.csv
│   ├── triples_mil_extracted.csv
│   └── sentence_attribution.jsonl
├── models/                         # CRF等模型
├── cypher/                         # Neo4j导入Cypher脚本
└── .venv/
```

---

## 三、运行方式

### 推荐运行命令

```bash
# 1. 完整构建
python run.py --mode full

# 2. 使用已有的curated中间层快速重放
python run.py --mode from_curated

# 3. 仅更新MIL关系并导出
python run.py --mode export_only

# 4. 单独研究MIL流程
python scripts/build_relation_bags.py
python scripts/build_ds_labels.py
python scripts/train_relation_pcnn.py --seed-type Person
python scripts/infer_relation_pcnn.py
python scripts/export_mil_triples.py
```

---

## 四、技术方案

### 方法概览

下表是各构建阶段所采用的核心方法与技术要点。详细说明见后续对应小节。


| 构建阶段    | 采用方法                                 | 核心技术要点                                         |
| ------- | ------------------------------------ | ---------------------------------------------- |
| 结构化知识底座 | Wikidata API 查询与声明聚合                 | 多 seed QID 并发查询；P31/P279 类型推断；初始知识图谱构建         |
| 多源文本采集  | 网页抓取 + PDF 文本提取 + 本地文档解析             | HTML 标签清洗与句子切分；PDF 指定页段抽取；统一 JSONL 样本格式        |
| 多中心主题归因 | 加权多特征相关性打分                           | anchor 命中 + trigger 覆盖 + 上下文弱一致性；句级证据筛选        |
| 命名实体识别  | 预训练 spaCy（主体语料）+ CRF（传记 PDF）         | zh/en_core_web_sm；字级特征工程；BIO 序列标注与解码           |
| 实体消歧    | 候选检索 + 局部打分 + Collective Linking     | Wikidata 标签匹配；Jaccard coherence score；贪心坐标上升解码 |
| 关系抽取    | 规则模板 + 远程监督（DS）+ PCNN-MIL            | 中英文模板槽位匹配；弱标签对齐；分段卷积 + 句级 Attention 聚合         |
| 事件抽取    | 触发词匹配 + 论元绑定                         | 四类事件触发词；复用 NER+EL 输出绑定论元；哈希 event_id 去重        |
| 图谱构建与导出 | GraphBuild 内存图 + 多源融合 + Neo4j CSV 入库 | facts/evidence 双层边；edge_id 去重；Cypher 脚本批量入库    |


---

### 4.1 系统总体架构

本项目以"多中心种子驱动的领域知识图谱构建"为核心任务，以 Wikidata 为结构化知识底座，以多源文本语料为证据输入，在统一的数据约定下完成实体识别、实体消歧、关系与事件抽取，并最终形成可入库的图谱数据产物。端到端构建流程如下：

```text
多源数据接入 → 多中心主题归因 → 实体识别 → 实体消歧 → 关系/事件抽取 → 多源融合 → 图谱构建与导出
```

---

### 4.2 多源数据采集与初始图谱构建

本节介绍图谱构建所使用的两类数据来源：基于 Wikidata 的结构化初始知识图谱，以及围绕多中心主题构建的多源文本语料。

#### 4.2.1 初始知识图谱（Wikidata 结构化层）

系统以 `sources/seed_entities.json` 中配置的三个核心 QID 为起点，通过 Wikidata API 批量获取实体声明与邻域信息，构建初始知识图谱。三个种子实体分别对应：

- **图灵（人物）**：Q7251（Alan Turing），人物相关声明及关联人物、机构
- **图灵机（理论概念）**：Q163310（Turing Machine），计算理论相关概念与文献
- **图灵奖（奖项实体）**：Q185667（Turing Award），历届得主与颁奖机构

Wikidata 声明中的实体类型由 `P31`（instance of）和 `P279`（subclass of）推断，作为后续实体节点标签的来源。这一结构化初始图谱同时为远程监督提供可对齐的 `(subject, P, object)` 三元组，作为关系抽取的弱标签来源。

#### 4.2.2 多源文本语料

为覆盖人物传记、理论背景、奖项历史等多种叙事视角，本项目从四类来源构建文本语料：

**Wikipedia 摘要**：通过 API 获取中英文 Wikipedia 的摘要文本，涵盖 Alan Turing、Turing machine、Turing Award 等核心条目。

**本地文章摘录**：以"图灵 / 图灵机 / 图灵奖"为关键词检索，从 Wikipedia 及相关学术页面中选取信息密度较高的段落，整理为带溯源字段（`cite_key / url / title`）的本地文本文件（`raw/excerpts/articles/`）。

**传记 PDF 节选**：《艾伦·图灵传：如谜的解谜者》第 1–50 页，经 `sources/pdf_sources.json` 配置后按指定页段抽取文本层，补充传记类叙事语料。

**网页增广**：通过 `sources/web_sources.json` 指定白名单 URL，使用 `scripts/fetch_web_sources.py` 抓取正文内容，覆盖以下页面：


| 来源站点                  | 页面内容                                                                     |
| --------------------- | ------------------------------------------------------------------------ |
| en.wikipedia.org      | Turing machine、Church–Turing thesis、Computability theory、Halting problem |
| mathworld.wolfram.com | Turing Machine（数学百科）                                                     |
| plato.stanford.edu    | Turing machine、Alan Turing、Turing test（斯坦福哲学百科）                          |
| amturing.acm.org      | ACM 图灵奖历届得主列表与简介                                                         |


#### 4.2.3 数据预处理

各来源文本经以下统一流程处理为句子级样本：网页与百科页面去除 HTML 标签、脚本块与多余空白后切句；本地文档补齐溯源字段后切句；PDF 按页段抽取文本层后切句。所有来源均执行 Unicode 规范化与空白压缩。

处理后的句子样本写入 `data/processed/sentences.jsonl`，每条记录包含 `source_id / source_url / citation_key / sentence_idx / sentence` 等字段，作为后续归因、NER、关系与事件抽取的统一上游。

#### 4.2.4 数据采集结果概况


| 指标                        | 数值        |
| ------------------------- | --------- |
| 文本来源数（unique `source_id`） | 33        |
| 句子样本总数（`sentences.jsonl`） | 8,726     |
| 句子语料体量                    | 约 3.20 MB |
| 主题归因记录数（`routing.jsonl`）  | 8,726     |
| 平均每句归因 seed 数             | 0.374     |


---

### 4.3 多中心主题归因

多源语料中，同一句子可能同时涉及图灵（人物）、图灵机（概念）和图灵奖（奖项）等不同中心的内容。若不加区分地将全部句子送入后续抽取流程，会造成不同中心的证据相互混入，增加噪声。因此，本项目在进入实体识别之前，对每条句子与每个种子中心计算相关性得分，依据得分决定该句是否作为对应中心的证据句参与后续处理。

#### 4.3.1 种子表示与评分机制

每个种子中心在 `sources/seed_entities.json` 中配置两类词表：

- **锚定词（anchors）**：中心实体的中英文常见指称形式（如"图灵""Alan Turing""Turing"），用于判断句子是否直接提及该中心。
- **触发词（triggers）**：与中心主题相关的领域术语（如"计算机""图灵完备""ACM 奖"），用于识别与中心语义相关但未直接指称的句子。

对句子 $s$ 与种子 $i$ 的相关性打分可表示为：

$$
\mathrm{rel}(s, i) = w_a \cdot \phi_{\mathrm{anchor}}(s, i) + w_t \cdot \phi_{\mathrm{trigger}}(s, i) + w_d \cdot \phi_{\mathrm{ctx}}(s, i) - \lambda \cdot \mathrm{penalty}(s, i)
$$

其中 $\phi_{\mathrm{anchor}}$ 统计锚定词命中情况，$\phi_{\mathrm{trigger}}$ 统计触发词覆盖情况，$\phi_{\mathrm{ctx}}$ 利用相邻句对同一主题的延续性给予弱加分，$\mathrm{penalty}$ 对与中心无关的噪声模式予以惩罚。各权重与阈值通过 `sources/attribution_config.json` 配置，支持调整后复现。

得分超过阈值的（sentence, seed）对被记录为一条归因记录，该句在后续处理中作为对应种子中心的证据输入。

#### 4.3.2 输出与结果概况

归因结果写入 `data/processed/routing.jsonl`，每条记录包含 `sentence_idx`、`seed_id`、`score` 及命中的 anchors / triggers 列表（`reasons` 字段），支持逐句审查与复现。


| 指标            | 数值                                     |
| ------------- | -------------------------------------- |
| 处理句子总数        | 8,726                                  |
| 平均每句归因 seed 数 | 0.374                                  |
| 实现模块          | `turing_kg/attribution/seed_router.py` |


归因得分低于阈值的句子不参与后续抽取，以此过滤大量与三个中心主题均无关的背景文本，在保留有效证据的同时控制后续 NER 和关系抽取的噪声规模。

---

### 4.4 命名实体识别

命名实体识别（NER）是从证据句中自动定位并标注实体 mention 的过程，其输出直接影响后续实体消歧与关系/事件抽取的质量。本项目采用**七类实体标注体系**（PER / ORG / LOC / DATE / TIME / MONEY / PERCENT），并对不同性质的语料采用不同的识别方法：以预训练 spaCy 模型处理主体网页与百科类语料，以条件随机场（CRF）处理领域传记类 PDF 文本。两套方法相互补充，且识别结果均携带可追溯的后端来源字段（`ner_backend`）。

#### 4.4.1 使用预训练 spaCy 模型识别（主体语料）

对于 Wikipedia 摘要、本地文章摘录、Web 增广页面等来源，本项目使用 spaCy 的预训练神经网络 NER 管线进行识别。

首先进行语言检测：对每段文本计算中文字符占比，若比例不低于 0.12 则判定为中文文本，据此分别选用 `zh_core_web_sm` 或 `en_core_web_sm` 模型进行处理，确保中英文语料下识别精度的稳定性。

spaCy 模型的原始输出类型标签并不直接对应本项目的七类体系，因此识别后需进行标签映射：PERSON 对应 PER，GPE / LOC / FAC 统一归并为 LOC，DATE、TIME、MONEY、PERCENT 直接对应同名类别，其余类型丢弃。与此同时，对于种子实体自身（即各 seed anchor 词）在文本中出现的 PERSON mention，系统将其从结果中排除，避免将研究主体混入候选对象实体。

在模型识别的基础上，系统还通过轻量规则进行补强：以正则匹配对中文日期（如"1912年6月23日"）、时间表达、货币金额、百分比等数值实体进行补充抽取；并对"英国""伦敦""剑桥""曼彻斯特"等领域高频地理专名进行字符串匹配补全，提升关系与事件抽取所需的 LOC 类别召回。

#### 4.4.2 基于条件随机场（CRF）的领域文本识别（传记 PDF 语料）

对于《艾伦·图灵传：如谜的解谜者》第 1–50 页的传记文本，由于其叙事风格与通用预训练语料差异显著（人名机构名密集、长句结构多），本项目采用**条件随机场（CRF）**模型进行识别，以实现对领域特征的更好适应。

CRF 的输入为**字级 token 序列**（character-level tokenization），将每个汉字、标点及数字视作独立 token，从而避免分词错误的级联影响。对每个 token，抽取以下手工特征：当前字本身、前后各一字（上下文窗口）、当前字与相邻字构成的二元组（bigram）、字的形状属性（汉字 / 数字 / 字母 / 标点）、是否位于句子边界（BOS / EOS）等。这些特征共同描述 token 在局部序列中的字形与位置特征，是 CRF 序列标注的经典特征工程范式。

模型以 **BIO 标注方案**输出预测结果，即每个 token 被标为 `B-PER`、`I-ORG`、`O` 等标签，随后通过标签解码将连续的 B/I 序列合并为完整的 mention span，并映射至七类标注体系。模型使用 `sklearn-crfsuite` 进行训练，训练数据由本项目从已有 mention 标注中自动构建的 BIO 格式语料（`data/ner_crf/train.conll` / `dev.conll`）提供，并以 seqeval entity-F1 进行验证，开发集 F1 约为 0.61。

#### 4.4.3 识别结果概况

在一次完整运行后，系统识别到的实体 mention 分布如下：


| 指标                                           | 数值                     |
| -------------------------------------------- | ---------------------- |
| 识别 mention 总数（`data/curated/mentions.jsonl`） | 3,127                  |
| 使用 spaCy 模型识别数量                              | 3,041（占 97.2%）         |
| 使用 CRF 模型识别数量                                | 86（占 2.8%，来自传记 PDF 语料） |


按实体类型分布如下：


| 实体类型         | 数量    | 说明              |
| ------------ | ----- | --------------- |
| DATE（日期）     | 1,189 | 传记与百科语料中密度最高的类型 |
| ORG（机构/组织）   | 583   | 大学、实验室、计算机协会等   |
| MONEY（货币金额）  | 432   | 奖金、资助、薪酬等       |
| LOC（地点）      | 337   | 出生地、工作地、研究机构所在地 |
| PER（人名）      | 323   | 关联学者、历史人物等      |
| PERCENT（百分比） | 205   | 科学统计类文本         |
| TIME（时间）     | 58    | 具体时刻            |


DATE 与 ORG 类别识别数量最多，与语料的主题特征（人物传记、机构背景、奖项记录）高度吻合。这些 mention 经实体消歧（4.5 节）后，有效实体进一步参与关系与事件抽取任务。

---

### 4.5 实体消歧

实体消歧（Entity Linking / Disambiguation）的任务是将 NER 阶段识别出的实体 mention 与知识库中的唯一实体条目对应，从而为关系抽取与图谱构建提供结构化、可溯源的实体标识。本项目以 Wikidata QID 作为统一实体标识符，对 PER / ORG / LOC 三类结构性实体执行链接，而 DATE / TIME / MONEY / PERCENT 等数值字面类 mention 不进行 KB 链接，以字面值形式保留。消歧过程分为两个阶段：局部候选检索与打分，以及基于上下文一致性的集体消歧解码。

#### 4.5.1 候选实体检索与局部打分

对每条 mention，首先通过实体检索生成 Top-K 候选 QID 列表。检索策略结合字符串相似度匹配与 Wikidata 标签查询：优先从预置的 `entity_map`（覆盖图灵、图灵机、图灵奖等高频核心实体的精确映射）中直接命中，以减少歧义；对于未命中的 mention，向 Wikidata 发起实体检索并返回候选集合。

候选打分采用局部得分（local score）机制，综合考量以下信号：mention 文本与候选标签的字符串相似度、候选实体的类型是否与 mention 的 NER 标签一致（类型约束过滤）、以及候选实体在知识库中的先验流行度（popularity prior）。局部打分结果写入 `data/curated/candidates.jsonl`，包含每条 mention 的候选列表及各候选得分，供后续消歧与误差分析使用。

#### 4.5.2 基于上下文一致性的集体消歧

单独依赖局部打分的消歧策略在歧义较高的场景下容易产生误链，尤其是同名人物或同名机构在不同上下文中具有不同含义时。为此，本项目在局部打分基础上引入**集体消歧机制**（Collective Entity Linking），通过考察句子窗口内多个 mention 候选实体之间的知识图谱结构一致性来辅助决策。

具体而言，对窗口大小为 2 句的上下文，系统为各候选 QID 提取其在 Wikidata 中的一跳邻域实体（仅保留 `coherence_props` 所指定的若干语义相关属性），并以 Jaccard 相似度度量两候选实体邻域集合的重叠程度，形成实体对之间的结构一致性得分（coherence score）。

最终解码采用**贪心坐标上升**策略：以局部得分为初始基底，在每轮迭代中依次为各 mention 选取使总目标（局部得分 + λ × coherence 增益）最大化的候选 QID，并持续迭代直至分配稳定。其中一致性权重 λ 默认设为 0.35，在不同实体密度文本中可调。该机制的核心思路是：窗口内共现的 mention 若在知识图谱结构上相互印证，则彼此的消歧结果可以互相提升置信度，从而有效抑制孤立 argmax 带来的误链传播。

#### 4.5.3 消歧结果概况

消歧结束后，所有 mention 的链接决策写入 `data/curated/resolved.jsonl`，包含 `chosen_qid`、分数分解及可读性 `reasons` 字段，支持后续审计与分析。结果概况如下：


| 指标                              | 数值             |
| ------------------------------- | -------------- |
| 参与消歧的 mention 总数                | 3,127          |
| 成功链接至 Wikidata QID 的 mention 数量 | 1,243（占 39.8%） |
| 保留为字面值（数值类，不做 KB 链接）            | 1,884（占 60.2%） |


结构性实体（PER / ORG / LOC）的链接覆盖情况：


| 实体类型                          | NER 识别数 | 成功链接数 | 链接率       |
| ----------------------------- | ------- | ----- | --------- |
| PER（人名）                       | 323     | 323   | 100%      |
| ORG（机构/组织）                    | 583     | 583   | 100%      |
| LOC（地点）                       | 337     | 337   | 100%      |
| DATE / TIME / MONEY / PERCENT | 1,884   | 0     | — （字面值保留） |


PER / ORG / LOC 三类实体全部参与候选检索与消歧，最终链接的 1,243 条 mention 将作为 `object_qid` 参与关系三元组构建与远程监督对齐（4.6 节）。

---

### 4.6 关系抽取

关系抽取的目标是从文本证据句中识别实体对之间的语义关系，并将其映射至以 Wikidata 属性（`P` 编号）为中心的关系类型体系，从而为知识图谱提供有来源依据的关系边。本项目综合运用两种互补的抽取路线：针对显式语言模式的**规则/模板匹配**，以及面向语料规模化的**远程监督与多实例学习**（Distant Supervision + PCNN-Attention）。两种路线的结果经去重与置信度排序后统一合并入图，并以 `provenance` 字段标记来源。

#### 4.6.1 基于规则模板的关系抽取

规则模板方法以人工设计的语言表达模式为核心，针对知识图谱领域中表达高度规律化的关系类型（如出生地、就职机构、国籍、重要著作等）设定中英文匹配模板。对于每个归因句（已通过 4.3 节的相关性评分筛选出的证据句），系统依次运行各关系的匹配器：若句子满足触发条件（如包含指定的动词短语、介词结构或上下文词汇），则从句内抽取对应位置的实体填入主客体槽位，生成关系三元组。

规则模板方法的优势在于精确率高且结果可解释：每条抽取到的三元组均有明确的模板名称与证据文本作为支撑，易于核查。其局限在于覆盖率受限于模板规模，难以穷举自然语言中的所有表达方式。因此，该方法主要用于高可信度的显式关系抽取，其结果在融合时具有较高优先级。

本次运行共通过规则模板抽取 **71 条关系三元组**，覆盖以下关系类型：


| 关系类型（Wikidata 属性）              | 三元组数量 |
| ------------------------------ | ----- |
| notable_work（代表作品，P800）        | 18    |
| place_of_birth（出生地，P19）        | 15    |
| country_of_citizenship（国籍，P27） | 14    |
| employer（就职机构，P108）            | 12    |
| work_location（工作地点，P937）       | 5     |
| country（所属国家，P17）              | 3     |
| location（所在地点，P276）            | 2     |
| educated_at（就读机构，P69）          | 2     |


#### 4.6.2 远程监督与多实例学习（DS + PCNN-Attention）

规则模板难以覆盖隐式或多样化表达的关系。为此，本项目引入**远程监督（Distant Supervision, DS）**策略，将 Wikidata 中已有的 `(subject_qid, P, object_qid)` 知识库声明与经实体消歧后的证据语料对齐：若某个归因句同时包含已链接的主体实体与客体实体，则将该句纳入该关系的训练样本（"bag"），并以知识库声明作为弱标签。

DS 方法在获得规模化训练数据的同时，也引入了大量噪声（知识库中两实体存在某关系，并不意味着每一条共现句都在表达该关系）。为此，本项目采用**多实例学习（Multi-Instance Learning, MIL）**对噪声进行抑制，以 **PCNN-Attention**（分段卷积神经网络 + 句级 Attention）为基础模型：

- **PCNN 编码**：将句子以主客体实体为分界点划分为左段、中段、右段三个分段，分别经卷积与最大池化后拼接，形成对实体间上下文的分段感知向量表示，有效捕捉位置敏感的关系线索。
- **句级 Attention 聚合**：在同一 bag 内的多条证据句上，以 softmax 权重对各句向量加权求和，形成 bag 级表示。权重较高的句子往往是对关系表达最显式的证据，从而实现对单句级 DS 噪声的软性过滤。
- **bag 级多标签分类**：在 bag 表示上做关系分类，预测该实体对可能对应的一个或多个 Wikidata 属性，并以置信度阈值控制输出精度。

模型按种子实体类型（Person / Award）与目标属性（P800、P27、P108 等）分别训练，以适应不同语义域的分布差异。

本次运行共通过 PCNN-MIL 推断 **82 条关系三元组**，覆盖以下关系类型：


| 关系类型（Wikidata 属性）              | 三元组数量 |
| ------------------------------ | ----- |
| notable_work（代表作品，P800）        | 29    |
| country_of_citizenship（国籍，P27） | 26    |
| employer（就职机构，P108）            | 18    |
| country（所属国家，P17）              | 5     |
| location（所在地点，P276）            | 4     |


#### 4.6.3 关系抽取结果概况

两路抽取方法的结果经去重与置信度排序合并，以 `provenance` 字段标记来源（`zh_template` / `pcnn_ds_`*），写入 `data/triples_extracted.csv` 与 `data/triples_mil_extracted.csv`，并最终融合进入图谱的关系边层（`layer: facts`）。


| 来源                     | 三元组数量 | 关系类型数 |
| ---------------------- | ----- | ----- |
| 规则/模板匹配（zh_template）   | 71    | 8     |
| PCNN-MIL 推断（pcnn_ds_*） | 82    | 5     |


句级共现（cooccurrence_linked）为两个已链接实体在同一归因句中的共现记录，是构造 DS 训练 bag 的原材料，本身不作为语义关系边入图。两路抽取所得的关系三元组（覆盖 8 类 Wikidata 属性）进入图谱关系层，与 Wikidata 结构层导入的 154 条事实边（4.2 节）共同构成图谱的关系骨架，为事件抽取与图谱查询提供关系依据（详见 4.7 节）。

---

### 4.7 事件抽取

事件抽取与关系抽取在研究对象上有所区别：关系抽取侧重实体对之间稳定的语义连接（如"就职于""出生于"），而事件抽取关注的是以特定动作或场景为中心的具体事件实例，包括事件类型、触发词、参与论元及时间信息。在图谱中，事件以独立节点的形式存在，并通过 `EXT_EVENT_ARG` 类型的边连接参与者实体，与关系层、事实层区分来源标记（`provenance: event_extraction`）。

本项目针对图灵相关语料的叙事特征，定义了四类事件类型：**AwardEvent**（颁奖/获奖）、**EmploymentEvent**（任职/受雇）、**EducationEvent**（求学/就读）和 **PublicationEvent**（提出/发表）。

#### 4.7.1 触发词检测与论元抽取

事件识别以**触发词匹配**为入口。对每条归因证据句，系统根据语言（中/英）选择对应的触发词集合进行扫描，命中触发词后记录触发片段及其在句中的位置区间。触发词的设计尽量贴合语料风格：

- AwardEvent 的英文触发词包括 `was awarded`、`won`、`recipient`、`prize` 等，中文触发词为"获奖""授予""颁发""得主"等；此外对"Awardee (Year)"格式的括号年份模式也做了正则匹配，以捕获常见的获奖者列表格式。
- EmploymentEvent 与 EducationEvent 共用一套探测器，以"worked at / joined / appointed"与"studied at / graduated from / attended"两组触发词区分两类事件。
- PublicationEvent 覆盖"proposed / introduced / published / paper / article"以及对应的中文词，用于抽取理论提出与成果发表类事件。

触发词命中后，系统从句内已有的 NER+EL 结果中绑定论元：PER 类 mention 对应发起者/参与者，ORG 类对应机构，DATE 类对应事件发生时间。论元绑定直接复用 4.4 节的实体识别输出，无需额外的论元识别模型。

#### 4.7.2 事件实例规范化与入图

同一事实（如图灵 1966 年获得图灵奖）可能在多个证据句中被触发，为避免重复节点，系统对事件实例进行规范化合并：以事件类型、核心论元（人物 QID、机构 QID、时间）的组合生成稳定的 `event_id`（哈希策略），同一事实的多条证据归并到同一节点下，并记录 `evidence_count`。最终，事件节点及其论元边写入 `GraphBuild`，通过 `ingest` 接口进入图谱，与关系层和事实层在来源字段上加以区分。

#### 4.7.3 事件抽取结果概况

本次运行共抽取 **247 条事件-论元关系边**，分布在四类事件之间：


| 事件类型             | 含义        | 论元边数量 |
| ---------------- | --------- | ----- |
| PublicationEvent | 理论提出与成果发表 | 116   |
| EmploymentEvent  | 任职与受雇     | 91    |
| AwardEvent       | 颁奖与获奖     | 36    |
| EducationEvent   | 求学与就读     | 4     |


参与论元的实体类型以 PER（110 条）和 ORG（44 条）为主，DATE 类论元（28 条）提供事件的时间定位。PublicationEvent 比例最高，与语料中图灵及相关学者学术贡献记录密集的主题特征相吻合；AwardEvent 覆盖了图灵奖历届得主的颁奖记录，是图谱中价值密度较高的结构化信息之一。

---

### 4.8 知识图谱构建与融合

前述各模块（Wikidata 结构层、文本语料、NER、实体消歧、关系抽取、事件抽取）的输出，最终需要汇聚为统一的图谱表示。本节介绍图谱的节点与边结构设计、多来源数据的融合策略，以及从内存图到 Neo4j 的持久化导出路径。

#### 4.8.1 节点体系与属性设计

图谱以 Wikidata QID 作为实体节点的主键（`:ID`），以确保来自不同模块的记录能够在同一节点上汇聚。每个节点携带 `:LABEL` 多标签体系，粗粒度标签（Entity）之下设领域语义标签（Person、Concept、Award、Organization、Location 等），由 Wikidata 类型推断或图谱规则赋予。

事件节点是本图谱的一类特殊节点。与关系边不同，事件被建模为独立节点（标签为 PublicationEvent / EmploymentEvent / AwardEvent / EducationEvent），携带 `event_type`、`evidence_count` 等属性，并通过 `EXT_EVENT_ARG` 类型的有向边连接参与者实体（人物、机构、时间字面值等）。这一设计使事件可以拥有任意数量的论元，并支持按事件类型独立查询，比将事件简化为二元关系边更具表达力。

本次运行图谱节点分布如下：


| 节点类型                                                                              | 数量      |
| --------------------------------------------------------------------------------- | ------- |
| Person（人物）                                                                        | 77      |
| Concept（概念，含数学概念、职业、名称等子类）                                                        | 77      |
| Event（事件，含 PublicationEvent 51、EmploymentEvent 42、AwardEvent 15、EducationEvent 4） | 112     |
| Literal（字面值，如日期、数量）                                                               | 46      |
| Organization（机构/组织）                                                               | 9       |
| Award（奖项）                                                                         | 5       |
| Paper / ScholarlyArticle（文献）                                                      | 7       |
| Location（城市/国家）                                                                   | 6       |
| **合计**                                                                            | **364** |


#### 4.8.2 边的分层结构与来源区分

图谱的边按语义层次分为两层：

**facts 层**（154 条）包含高置信度的结构化声明边，来源为 Wikidata 导入（`provenance: wikidata` / `wikidata_incoming`）。边类型以 `WIKI_P{属性编号}` 命名（如 `WIKI_P166` 表示"获得奖项"、`WIKI_P800` 表示"代表作"），与 Wikidata 属性体系直接对应，是图谱的事实骨架。

**evidence 层**（250 条）包含从文本中抽取的证据性边，来源为事件抽取（`event_extraction`，247 条 `EXT_EVENT_ARG` 边）和规则/模型关系抽取（`text_re`，3 条）。这类边携带 `snippet`（证据原句）、`citation_key`（文献索引）、`source_url` 等字段，支持溯源核查。

每条边均分配一个由来源、实体对、属性、证据句等字段拼接哈希生成的 `edge_id`，保证多次运行结果的可重复去重，同时允许同一实体对在不同证据句下存在多条 evidence 边（"同对多证"）。

#### 4.8.3 多源融合与去重策略

各模块的输出在写入图谱时，按以下原则进行融合：

对于同一属性的关系边（相同 `起点 QID`、`终点 QID`、`prop_id`），若来源为 Wikidata（facts 层），则视为高置信声明优先保留；若来源为规则模板或 MIL 模型（text_re），则补充写入 evidence 层，而非覆盖 facts 层，以保留多来源的交叉印证价值。事件边因具有独立的 `event_id` 主键，不与关系边发生冲突。

在导出前，系统通过行级合并逻辑（`turing_kg/attribution/triple_merge.py`）对来源相同、内容重复的三元组进行去重，最终形成无冗余的关系视图写入 `data/relationships.csv`。

#### 4.8.4 持久化导出与 Neo4j 入库

图谱在内存中以 `GraphBuild` 自定义图结构维护，节点与边均以字典方式存储，支持增量写入与 `edge_id` 级去重。构建完成后，通过导出接口（`turing_kg/io/export_io.py`）序列化为 Neo4j Bulk Import 格式的 CSV 文件：节点表（`data/nodes.csv`）和关系表（`data/relationships.csv`）分别对应 `neo4j-admin import` 所需的格式规范。

入库通过 `cypher/load_csv.cypher` 脚本执行，也可通过 `turing_kg/neo4j_loader.py` 以 Python Driver 方式写入，两者消费同一份 CSV，语义等价。入库后，图谱支持 Cypher 查询，可按实体类型、关系类型、事件类型、来源层（facts / evidence）等维度灵活检索。

**图谱规模总览**


| 维度                     | 数值  |
| ---------------------- | --- |
| 节点总数                   | 364 |
| 边总数                    | 404 |
| facts 层边（Wikidata 结构化） | 154 |
| evidence 层边（文本抽取）      | 250 |
| 覆盖 Wikidata 属性种类       | 12+ |


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
- **最终导出**：nodes.csv（364 个节点）、relationships.csv、triples_extracted.csv、triples_mil_extracted.csv
- **Neo4j导入**：cypher/load_csv.cypher

注：中间层主要采用 `JSONL`，最终导出采用 `CSV`。

---

## 七、开发进度

**当前已完成的工作**：

- 多中心种子配置驱动的全链路图谱构建（`run.py --mode full`）
- Wikidata 结构化知识底座的自动构建与类型推断
- 多源文本采集与统一句子级样本组织
- 多中心主题归因与证据句筛选（`routing.jsonl`）
- 双后端命名实体识别（spaCy 预训练 + CRF 领域模型）
- 集体一致性实体消歧（Collective Linking）
- 规则模板关系抽取 + 远程监督 + PCNN-MIL-Attention 推断
- 四类事件抽取（触发词 + 论元绑定 + 哈希规范化去重）
- 图谱多源融合与 Neo4j 可导入 CSV 导出
- `from_curated` 快速重放模式，支持从已审计中间层复现实验

