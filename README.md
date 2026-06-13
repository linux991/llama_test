# Llama 3 Horror/Thriller QLoRA Experiment

This project is a research prototype for a master's thesis. The goal is not to build a production-ready horror text generator, but to compare baseline and QLoRA-adapted Llama 3 8B Instruct on genre-specific generation in English and Russian.

## Research Goal

The practical experiment compares how well one base model generates horror/thriller prose before and after small QLoRA adaptation on genre corpora.

Main model:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

## Experimental Design


| Language | Baseline            | Fine-tuned                            |
| -------- | ------------------- | ------------------------------------- |
| English  | Llama 3 8B Instruct | Llama 3 8B Instruct + EN LoRA adapter |
| Russian  | Llama 3 8B Instruct | Llama 3 8B Instruct + RU LoRA adapter |


Qwen, detector-only runs, and older notebooks are kept only as legacy material. They are not part of the main thesis line.

## Repository Layout

```text
configs/                 # EN/RU experiment and generation settings
data/processed/en/       # normalized EN train/eval JSONL payloads
data/processed/ru/       # normalized RU train/eval JSONL payloads
notebooks/               # thesis-facing Colab notebooks
src/data/                # payload conversion scripts
src/generation/          # fixed prompts and generation helpers
src/evaluation/          # stylometry, statistics, manual eval templates
outputs/                 # generation and evaluation CSVs
adapters/                # EN/RU LoRA adapter directories
archive/old_experiments/ # legacy notebooks and old experiment material
reports/audit/           # repository audit notes
```

## Data Preparation

The reproducible experiment uses three independent splits for each language:

```text
500 training chunks
100 validation chunks
100 test chunks

```

The split is performed at story level. Chunks from the same source story cannot appear in different subsets.

Create the English and Russian datasets:

```bash
python src/data/prepare_three_way_split.py \
  --language all \
  --train-size 500 \
  --validation-size 100 \
  --test-size 100 \
  --seed 42

```

Expected outputs:

```text
data/processed/en/train_payload.jsonl
data/processed/en/validation_payload.jsonl
data/processed/en/test_payload.jsonl
data/processed/en/metadata.json

data/processed/ru/train_payload.jsonl
data/processed/ru/validation_payload.jsonl
data/processed/ru/test_payload.jsonl
data/processed/ru/metadata.json

```

The metadata files record the number of rows and unique stories and confirm that story overlap between train, validation and test is zero.

## Baseline Generation

Baseline outputs must be generated only from the held-out test subsets.

Each test row uses:

- its own system message;
- its own individual prompt;
- a deterministic sample seed;
- the same decoding parameters later used with the LoRA adapter.

English baseline:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_baseline \
  --language en \
  --test-jsonl data/processed/en/test_payload.jsonl \
  --output-csv outputs/en/baseline_generations.csv \
  --max-samples 100 \
  --seed 42

```

Russian baseline:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_baseline \
  --language ru \
  --test-jsonl data/processed/ru/test_payload.jsonl \
  --output-csv outputs/ru/baseline_generations.csv \
  --max-samples 100 \
  --seed 42

```

Required outputs:

```text
outputs/en/baseline_generations.csv
outputs/ru/baseline_generations.csv

```

## QLoRA Fine-Tuning

The base model is loaded in 4-bit format using:

```text
quantization type = NF4
compute dtype = float16
double quantization = enabled

```

LoRA and training parameters:

```text
LoRA r = 8
LoRA alpha = 16
LoRA dropout = 0.05
epochs = 1
learning rate = 1e-5
batch size = 1
gradient accumulation = 8
effective batch size = 8
max sequence length = 1024
target modules = q_proj, v_proj
logging steps = 10
validation steps = 10
checkpoint steps = 10

```

English training:

```bash
PYTHONPATH=$PWD python -m src.training.train_qlora \
  --language en \
  --train-jsonl data/processed/en/train_payload.jsonl \
  --validation-jsonl data/processed/en/validation_payload.jsonl \
  --adapter-dir adapters/en_lora_adapter \
  --artifacts-dir outputs/en/training \
  --max-train-samples 500 \
  --max-validation-samples 100 \
  --logging-steps 10 \
  --eval-steps 10 \
  --save-steps 10 \
  --seed 42

```

Russian training:

```bash
PYTHONPATH=$PWD python -m src.training.train_qlora \
  --language ru \
  --train-jsonl data/processed/ru/train_payload.jsonl \
  --validation-jsonl data/processed/ru/validation_payload.jsonl \
  --adapter-dir adapters/ru_lora_adapter \
  --artifacts-dir outputs/ru/training \
  --max-train-samples 500 \
  --max-validation-samples 100 \
  --logging-steps 10 \
  --eval-steps 10 \
  --save-steps 10 \
  --seed 42

```

The adapters are saved to:

```text
adapters/en_lora_adapter/
adapters/ru_lora_adapter/

```

Training artifacts are saved separately for each language:

```text
outputs/en/training/
outputs/ru/training/

```

Each training directory contains:

```text
environment.json
run_config.json
training_history.csv
training_metrics.json
trainer_state.json
loss_curve.png

```

The artifacts include training loss, validation loss, validation perplexity, next-token accuracy, runtime, GPU information and library versions.

## Fine-Tuned Generation

Fine-tuned outputs must use the same test rows, individual prompts, decoding parameters and sample seeds as the baseline outputs.

English:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_with_adapter \
  --language en \
  --test-jsonl data/processed/en/test_payload.jsonl \
  --adapter-dir adapters/en_lora_adapter \
  --output-csv outputs/en/finetuned_generations.csv \
  --max-samples 100 \
  --seed 42

```

Russian:

```bash
PYTHONPATH=$PWD python -m src.generation.generate_with_adapter \
  --language ru \
  --test-jsonl data/processed/ru/test_payload.jsonl \
  --adapter-dir adapters/ru_lora_adapter \
  --output-csv outputs/ru/finetuned_generations.csv \
  --max-samples 100 \
  --seed 42

```

Required outputs:

```text
outputs/en/finetuned_generations.csv
outputs/ru/finetuned_generations.csv

```

## Evaluation

The evaluation pipeline contains four layers:

1. Stylometric feature extraction.
2. Paired Wilcoxon signed-rank tests.
3. Paired rank-biserial effect sizes.
4. Manual and qualitative evaluation.

Before calculating metrics, the evaluation script verifies that baseline and fine-tuned files contain:

- identical test IDs;
- identical prompts;
- identical system messages;
- identical reference texts;
- identical sample seeds;
- identical decoding settings.

English evaluation:

```bash
PYTHONPATH=$PWD python -m src.evaluation.run_stylometry \
  --language en \
  --test-jsonl data/processed/en/test_payload.jsonl \
  --baseline-csv outputs/en/baseline_generations.csv \
  --finetuned-csv outputs/en/finetuned_generations.csv \
  --features-csv outputs/en/stylometric_features.csv \
  --summary-csv outputs/en/metric_summary.csv \
  --validation-report outputs/en/comparison_validation.json

```

Russian evaluation:

```bash
PYTHONPATH=$PWD python -m src.evaluation.run_stylometry \
  --language ru \
  --test-jsonl data/processed/ru/test_payload.jsonl \
  --baseline-csv outputs/ru/baseline_generations.csv \
  --finetuned-csv outputs/ru/finetuned_generations.csv \
  --features-csv outputs/ru/stylometric_features.csv \
  --summary-csv outputs/ru/metric_summary.csv \
  --validation-report outputs/ru/comparison_validation.json

```

Expected evaluation outputs:

```text
outputs/en/stylometric_features.csv
outputs/en/metric_summary.csv
outputs/en/comparison_validation.json

outputs/ru/stylometric_features.csv
outputs/ru/metric_summary.csv
outputs/ru/comparison_validation.json

```

Create manual evaluation templates after the final generation files are available:

```bash
PYTHONPATH=$PWD python -m src.evaluation.manual_eval_template

```

The test subsets must never be used during QLoRA training or validation.

## Known Limitations

Llama 3 8B Instruct is stronger in English than Russian. Russian generation may be less stable. Small-corpus QLoRA may not improve every metric. Automatic stylometry does not fully measure literary quality, so manual expert evaluation is required. The experiment is constrained by Colab/GPU resources.