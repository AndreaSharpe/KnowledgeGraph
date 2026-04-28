## 关系抽取 gold 评估闭环（经典做法）

### 目的
不要用“三元组条数”判断效果；要用 **gold(dev/test)** 的指标（accuracy/macro-F1/每类PRF）驱动阈值与数据改进。

### 文件约定
- **Gold**：`data/gold/re_gold_{seed_type}_{split}.jsonl`
  - `seed_type ∈ {Person, Concept, Award}`
  - `split ∈ {dev, test}`
- **Preds**：`data/curated/re_predictions_pcnn_{seed_type}.jsonl`（或 `re_predictions.jsonl`）
- **报告**：`data/analysis/re_eval_{seed_type}_{split}.json`
- **阈值配置**：`sources/relation_thresholds.json`（v2：`by_seed_type`）

### Step 1：抽样生成 gold 文件（需要人工标注）

```bash
.\.venv\Scripts\python.exe scripts\sample_re_gold.py --seed-type Person --split dev --n 60
```

打开 `data/gold/re_gold_Person_dev.jsonl`，把每条的 `label` 填成：\n- `NA`（无关系）\n- 或具体关系的 `Pxxx`（例如 `P27`、`P19`、`P108`…）

### Step 2：在 dev 上评估并自动搜 best_tau

```bash
.\.venv\Scripts\python.exe scripts\eval_re_gold.py --seed-type Person --split dev
```

输出报告：`data/analysis/re_eval_Person_dev.json`，重点看：\n- `best_tau`\n- `report_at_best_tau.macro_f1`\n- `per_label`（每类 PRF 与 support）

### Step 3：把 dev 的 best_tau 写回阈值配置（让导出/入图自动生效）

```bash
.\.venv\Scripts\python.exe scripts\tune_re_thresholds_from_gold.py --seed-type Person --write
```

之后你无需重跑推理，只要：\n- 重新导出 `triples_mil_extracted.csv` 或重新 `run.py --mode export_only`\n- 你就会得到按新阈值过滤后的三元组与图中证据边规模

### Step 4：在 test 上报告（不要再调参）

```bash
.\.venv\Scripts\python.exe scripts\eval_re_gold.py --seed-type Person --split test --tau <dev_best_tau>
```

### 迭代准则（每次改动后要跑什么）
- **改 NER/EL（entity_map / 规则 / rebuild script）**：重建 curated → 重建 bags/ds_labels → 重新训练/推理 → 跑 dev 评估\n- **只改阈值**：不重跑推理；直接 `tune_re_thresholds_from_gold.py` 写阈值 + 重新 export/ingest\n+
