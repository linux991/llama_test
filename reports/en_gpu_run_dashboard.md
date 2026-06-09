# EN Llama 3 QLoRA Run Dashboard

Date prepared: 2026-06-09  
Run folder: `new_gpu_results_only/`

## 1. Executive Summary

English GPU run completed successfully. The repository now contains:

- `100` English baseline generations.
- `25` English fine-tuned generations.
- A saved English LoRA adapter.
- Stylometric feature table.
- Metric summary table.

Technical outcome: the QLoRA adapter was trained and can be loaded for inference. Fine-tuned generation produces valid English horror/thriller prose.

Research outcome: the current fine-tuned sample is useful as a diagnostic result, but the measurable automatic stylometric shift from baseline is weak. For a stronger final thesis comparison, generate at least `50` fine-tuned samples, ideally matching the baseline sample count.

## 2. Artifact Inventory

Main result files:

| Artifact | Path | Notes |
| --- | --- | --- |
| Baseline generations | `new_gpu_results_only/outputs/en/baseline_generations.csv` | 100 rows |
| Fine-tuned generations | `new_gpu_results_only/outputs/en/finetuned_generations.csv` | 25 rows |
| Stylometric features | `new_gpu_results_only/outputs/en/stylometric_features.csv` | 225 rows |
| Metric summary | `new_gpu_results_only/outputs/en/metric_summary.csv` | 36 comparison rows |
| Final LoRA adapter | `new_gpu_results_only/adapters/en_lora_adapter/` | Use this for inference |
| Training checkpoint | `new_gpu_results_only/adapters/en_lora_adapter_checkpoints/checkpoint-63/` | Includes optimizer/scheduler state |

Approximate folder sizes:

| Folder | Size |
| --- | ---: |
| `new_gpu_results_only/` | 88 MB |
| `new_gpu_results_only/adapters/` | 87 MB |
| `new_gpu_results_only/adapters/en_lora_adapter/` | 30 MB |
| `new_gpu_results_only/adapters/en_lora_adapter_checkpoints/` | 57 MB |
| `new_gpu_results_only/outputs/en/` | 692 KB |

For dissertation analysis, the final adapter directory is enough. The checkpoint directory is useful for traceability, but not necessary for generation.

## 3. Model And Training Setup

Base model:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

Adapter type:

```text
LoRA / QLoRA-style 4-bit base loading
```

Adapter config from `adapter_config.json`:

| Parameter | Value |
| --- | --- |
| PEFT type | `LORA` |
| Task type | `CAUSAL_LM` |
| LoRA rank `r` | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj`, `v_proj` |
| Bias | `none` |
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |

Trainable parameter report:

```text
trainable params: 3,407,872
all params: 8,033,669,120
trainable%: 0.0424
```

Training data:

| Split | Rows |
| --- | ---: |
| Train | 500 |
| Eval | 100 |

Training state:

| Field | Value |
| --- | ---: |
| Epochs | 1.0 |
| Global steps | 63 |
| Train batch size | 1 |
| Eval loss | 3.4337 |
| Eval runtime | 71.79 s |
| Eval samples/sec | 1.393 |

Loss log:

| Step | Epoch | Loss | Grad norm | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.16 | 3.5448 | 1.1121 | 8.571e-06 |
| 20 | 0.32 | 3.5459 | 1.2029 | 6.984e-06 |
| 30 | 0.48 | 3.4789 | 1.3367 | 5.397e-06 |
| 40 | 0.64 | 3.4775 | 1.3279 | 3.810e-06 |
| 50 | 0.80 | 3.3842 | 1.2779 | 2.222e-06 |
| 60 | 0.96 | 3.4209 | 1.3880 | 6.349e-07 |

Interpretation: training is technically stable. Loss decreases modestly, which is expected for a small 1-epoch LoRA run on 500 chunks.

## 4. Generation Setup

Generation settings:

| Parameter | Value |
| --- | --- |
| Temperature | 0.8 |
| Top-p | 0.9 |
| Max new tokens | 300 |
| Seed | 42 |
| Prompt source | Fixed EN prompt |

Fixed EN prompt:

```text
Write a short literary scene in the horror/thriller genre.
Focus on atmosphere, suspense, sensory details, and psychological tension.
Do not explain the horror directly. Show a coherent scene.
```

Important caveat: because all generations use the same fixed prompt, the model often falls into similar horror openings. This is visible in both baseline and fine-tuned outputs.

## 5. Dataset And Output Counts

| Condition | Rows in analysis |
| --- | ---: |
| Human eval fragments | 100 |
| Baseline generations | 100 |
| Fine-tuned generations | 25 |
| Total stylometric rows | 225 |

ID alignment:

| Check | Result |
| --- | --- |
| Baseline unique IDs | 100 |
| Fine-tuned unique IDs | 25 |
| Fine-tuned IDs present in baseline | 25 / 25 |
| Fine-tuned order matches first matching baseline IDs | Yes |

Important statistics caveat:

`metric_summary.csv` currently compares `100` baseline rows against `25` fine-tuned rows. This is usable for diagnostics, but a final thesis comparison should use balanced groups:

- either generate `50-100` fine-tuned rows;
- or create a matched `25 baseline vs 25 fine-tuned` analysis using the same IDs.

## 6. Generation Length Diagnostics

Baseline:

| Metric | Value |
| --- | ---: |
| Rows | 100 |
| Empty generations | 0 |
| Unique generated texts | 100 |
| Char count min / median / mean / max | 1294 / 1379 / 1379.2 / 1455 |
| Word count min / median / mean / max | 222 / 238 / 238.1 / 248 |

Fine-tuned:

| Metric | Value |
| --- | ---: |
| Rows | 25 |
| Empty generations | 0 |
| Unique generated texts | 25 |
| Char count min / median / mean / max | 1325 / 1385 / 1381.1 / 1427 |
| Word count min / median / mean / max | 228 / 238 / 237.8 / 248 |

Interpretation: both baseline and fine-tuned outputs have stable length and no empty outputs. Fine-tuned generation did not collapse.

## 7. Stylometric Means By Condition

| Feature | Human, n=100 | Baseline, n=100 | Fine-tuned, n=25 |
| --- | ---: | ---: | ---: |
| `char_count` | 1440.84 | 1379.21 | 1381.08 |
| `word_count` | 272.60 | 238.43 | 237.96 |
| `type_token_ratio` | 0.5735 | 0.6552 | 0.6624 |
| `fear_word_rate` | 0.0061 | 0.0107 | 0.0098 |
| `suspense_word_rate` | 0.0055 | 0.0134 | 0.0148 |
| `repetition_score` | 0.4265 | 0.3448 | 0.3376 |

Interpretation:

- Human eval fragments are longer than generated outputs.
- Generated outputs have higher type-token ratio and lower repetition score, likely because they are shorter and more polished.
- Baseline and fine-tuned outputs are very close on the current automatic features.
- Fine-tuned outputs show slightly higher `suspense_word_rate` and slightly lower `repetition_score`, but the effect is small.

## 8. Statistical Summary

### 8.1 Human vs Baseline

Selected rows from `metric_summary.csv`:

| Feature | Human mean | Baseline mean | p-value | Rank-biserial |
| --- | ---: | ---: | ---: | ---: |
| `char_count` | 1440.84 | 1379.21 | 3.54e-20 | 0.7532 |
| `word_count` | 272.60 | 238.43 | 1.83e-27 | 0.8885 |
| `sentence_count` | 20.73 | 14.92 | 7.95e-13 | 0.5844 |
| `type_token_ratio` | 0.5735 | 0.6552 | 2.64e-29 | -0.9200 |
| `fear_word_rate` | 0.0061 | 0.0107 | 2.29e-10 | -0.5178 |
| `suspense_word_rate` | 0.0055 | 0.0134 | 5.54e-18 | -0.7050 |
| `repetition_score` | 0.4265 | 0.3448 | 2.64e-29 | 0.9200 |

Interpretation: human and baseline distributions differ strongly across several stylometric features. However, some differences are length-related and should not be interpreted as direct literary quality differences.

### 8.2 Human vs Fine-tuned

Selected rows:

| Feature | Human mean | Fine-tuned mean | p-value | Rank-biserial |
| --- | ---: | ---: | ---: | ---: |
| `char_count` | 1440.84 | 1381.08 | 3.34e-09 | 0.7668 |
| `word_count` | 272.60 | 237.96 | 6.94e-12 | 0.8892 |
| `sentence_count` | 20.73 | 15.12 | 7.61e-06 | 0.5792 |
| `type_token_ratio` | 0.5735 | 0.6624 | 8.27e-13 | -0.9280 |
| `fear_word_rate` | 0.0061 | 0.0098 | available in CSV | direction: fine-tuned higher |
| `suspense_word_rate` | 0.0055 | 0.0148 | available in CSV | direction: fine-tuned higher |

Interpretation: fine-tuned outputs remain distinct from human fragments in a similar way to baseline outputs.

### 8.3 Baseline vs Fine-tuned

Unbalanced comparison from `metric_summary.csv`: `baseline n=100` vs `fine-tuned n=25`.

| Feature | Baseline mean | Fine-tuned mean | p-value | Rank-biserial |
| --- | ---: | ---: | ---: | ---: |
| `char_count` | 1379.21 | 1381.08 | 0.7435 | -0.0428 |
| `word_count` | 238.43 | 237.96 | 0.6250 | 0.0636 |
| `sentence_count` | 14.92 | 15.12 | 0.5408 | -0.0788 |
| `type_token_ratio` | 0.6552 | 0.6624 | 0.1640 | -0.1808 |
| `fear_word_rate` | 0.0107 | 0.0098 | 0.5089 | 0.0860 |
| `suspense_word_rate` | 0.0134 | 0.0148 | 0.2855 | -0.1388 |
| `cliche_count` | 0.21 | 0.24 | 0.4917 | -0.0604 |
| `repetition_score` | 0.3448 | 0.3376 | 0.1640 | 0.1808 |

Interpretation: no strong automatic stylometric separation between baseline and fine-tuned outputs is visible in the current English run.

### 8.4 Matched 25 vs 25 Diagnostic

A local matched check was run by comparing the 25 fine-tuned IDs against the same 25 baseline IDs.

| Feature | Baseline 25 mean | Fine-tuned 25 mean | Rank-biserial |
| --- | ---: | ---: | ---: |
| `char_count` | 1382.76 | 1381.08 | -0.0208 |
| `word_count` | 237.32 | 237.96 | -0.0880 |
| `sentence_count` | 14.96 | 15.12 | -0.0224 |
| `type_token_ratio` | 0.6632 | 0.6624 | 0.0288 |
| `fear_word_rate` | 0.0118 | 0.0098 | 0.1984 |
| `suspense_word_rate` | 0.0149 | 0.0148 | 0.0400 |
| `cliche_count` | 0.12 | 0.24 | -0.1200 |
| `repetition_score` | 0.3368 | 0.3376 | -0.0288 |

Interpretation: matched analysis confirms that the fine-tuned adapter changes some text content, but does not substantially move the measured stylometric distribution.

## 9. Qualitative Diagnostics

### 9.1 Fine-tuned differs from baseline, but often only locally

Exact text match check:

| Check | Result |
| --- | ---: |
| Fine-tuned rows | 25 |
| Exact matches with baseline on same ID | 2 |
| Different from baseline on same ID | 23 |

This means the adapter is active and generation is not simply reusing the baseline CSV. However, many changes are local phrase-level substitutions rather than a major style shift.

Example difference, shortened:

Baseline:

```text
Cobwebs clung to my face, sticky threads that seemed to wrap around my skin like tiny, suffocating arms.
Every step echoed through the empty halls, a morbid cadence...
```

Fine-tuned:

```text
Cobwebs clung to my face, sticky threads that pulled at my skin like tiny fingers.
Every step echoed through the empty halls, a lonely cadence...
```

### 9.2 Strong prompt-induced repetition

Common starts:

| Condition | Frequent opening pattern |
| --- | --- |
| Baseline | `The old mansion loomed before me, its turrets...` |
| Baseline | `The old mansion loomed before her, its turrets...` |
| Fine-tuned | `The old mansion loomed before me, its turrets...` |
| Fine-tuned | `The old mansion loomed before her, its turrets...` |

Keyword repetition:

| Diagnostic | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Contains `old mansion` | 68 / 100 | 20 / 25 |
| Contains `asylum` | 5 / 100 | 2 / 25 |
| Contains `Emma` | 27 / 100 | 9 / 25 |

Interpretation: the fixed prompt encourages a narrow cluster of horror tropes. This should be reported as a limitation of the generation protocol.

## 10. What This Means For The Thesis

The English experiment currently supports the following cautious claims:

1. A small QLoRA adapter for `meta-llama/Meta-Llama-3-8B-Instruct` was successfully trained on an English horror/thriller corpus.
2. The adapter can be loaded and used for generation.
3. Fine-tuned outputs are valid and mostly differ from baseline outputs on the same prompts.
4. In this small run, automatic stylometric metrics do not show a strong baseline-to-fine-tuned shift.
5. Both baseline and fine-tuned outputs differ strongly from human fragments in length-related and lexical-distribution metrics.
6. The fixed generic prompt leads to repetitive motifs and openings, especially `old mansion` scenarios.

Recommended dissertation framing:

> The English QLoRA adaptation was technically successful and produced coherent genre outputs. However, under the current small-data, one-epoch setup and fixed-prompt generation protocol, automatic stylometric measures did not demonstrate a large shift from the baseline model. This suggests that the adapter effect is subtle and that qualitative evaluation and prompt-design limitations should be discussed alongside quantitative results.

## 11. Limitations To Mention

Use these in the dissertation discussion:

- Fine-tuned generation currently has only `25` samples, while baseline has `100`.
- Automatic metrics are sensitive to text length; human fragments are longer than generated outputs.
- The fixed prompt causes repeated horror cliches and repeated settings.
- QLoRA was intentionally minimal: `500` train chunks, `1` epoch, `q_proj/v_proj` only.
- A weak metric shift does not mean the adapter failed; it may mean the adaptation was too small to move coarse stylometric features.
- Manual evaluation is needed to assess genre fit, suspense, coherence, and originality.

## 12. Recommended Next Actions

Priority order:

1. Generate at least `50` fine-tuned English samples, ideally `100`, using the same eval prompts.
2. Recompute `stylometric_features.csv` and `metric_summary.csv` after balancing baseline/fine-tuned sizes.
3. Run the same pipeline for Russian.
4. Create manual evaluation samples:

```bash
PYTHONPATH=$PWD python -m src.evaluation.manual_eval_template
```

5. In the thesis, use automatic metrics as descriptive/statistical evidence, not as the only quality criterion.

## 13. Files To Send To External ChatGPT

Send these files or paste their key content:

```text
reports/en_gpu_run_dashboard.md
new_gpu_results_only/outputs/en/baseline_generations.csv
new_gpu_results_only/outputs/en/finetuned_generations.csv
new_gpu_results_only/outputs/en/stylometric_features.csv
new_gpu_results_only/outputs/en/metric_summary.csv
new_gpu_results_only/adapters/en_lora_adapter/adapter_config.json
new_gpu_results_only/adapters/en_lora_adapter_checkpoints/checkpoint-63/trainer_state.json
```

If file upload is limited, send only:

```text
reports/en_gpu_run_dashboard.md
new_gpu_results_only/outputs/en/metric_summary.csv
```

## 14. One-Paragraph Summary For External ChatGPT

English Llama 3 8B Instruct QLoRA run completed successfully. The model was adapted with a LoRA adapter (`r=8`, `alpha=16`, dropout `0.05`, target modules `q_proj` and `v_proj`) on 500 English horror/thriller training chunks for 1 epoch. The final training checkpoint reached 63 steps with eval loss about 3.43. The run produced 100 baseline generations and 25 fine-tuned generations using the same fixed horror/thriller prompt. Fine-tuned outputs were valid and 23 of 25 differed from the matching baseline generations, showing that the adapter was active. However, automatic stylometric metrics show little separation between baseline and fine-tuned outputs; the strongest differences remain between human fragments and generated texts. A major qualitative limitation is prompt-induced repetition: both baseline and fine-tuned generations frequently start with “The old mansion...” and reuse common horror tropes. The result should be framed as a successful technical adaptation with modest measurable stylistic effect under a small-data, one-epoch, fixed-prompt protocol.
