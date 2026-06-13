#!/usr/bin/env python3
"""Train and evaluate a QLoRA adapter for the thesis experiment.

The script:

1. Loads Llama 3 8B Instruct in 4-bit NF4 format.
2. Trains a LoRA adapter on a training JSONL dataset.
3. Uses a separate validation JSONL dataset during training.
4. Logs training and validation loss every N optimizer steps.
5. Calculates validation perplexity and next-token accuracy.
6. Saves the adapter, tokenizer, metrics, environment information,
   training history and a loss curve.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.lora_config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    TARGET_MODULES,
)
from src.utils.logging import StepTimer, get_logger, log_kv, log_stage


LOGGER = get_logger("train_qlora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        choices=["en", "ru"],
        required=True,
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )

    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            prompt = str(row.get("prompt", "")).strip()
            text = str(row.get("text", "")).strip()

            if not prompt:
                raise ValueError(
                    f"Empty prompt in {path}, line {line_number}."
                )

            if not text:
                raise ValueError(
                    f"Empty target text in {path}, line {line_number}."
                )

            rows.append(row)

            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No usable rows found in {path}.")

    return rows


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    system_message = str(row.get("system", "")).strip()

    if system_message:
        messages.append(
            {
                "role": "system",
                "content": system_message,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": str(row["prompt"]).strip(),
        }
    )

    return messages


def tokenize_rows(
    tokenizer,
    rows: list[dict[str, Any]],
    max_seq_length: int,
) -> Dataset:
    """Tokenize chat examples and mask non-assistant tokens in labels."""

    tokenized_rows: list[dict[str, list[int]]] = []

    for row in rows:
        prompt_messages = build_prompt_messages(row)

        full_messages = [
            *prompt_messages,
            {
                "role": "assistant",
                "content": str(row["text"]).strip(),
            },
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

        input_ids = list(full_tokens["input_ids"])
        attention_mask = list(full_tokens["attention_mask"])
        labels = input_ids.copy()

        prompt_length = min(
            len(prompt_tokens["input_ids"]),
            len(labels),
        )

        labels[:prompt_length] = [-100] * prompt_length

        if not any(label != -100 for label in labels):
            LOGGER.warning(
                "Skipped example %s because its assistant response "
                "was fully truncated.",
                row.get("id", "unknown"),
            )
            continue

        tokenized_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if not tokenized_rows:
        raise ValueError(
            "All examples were removed during tokenization. "
            "Increase max sequence length or inspect the input data."
        )

    return Dataset.from_list(tokenized_rows)


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "transformers_version": package_version("transformers"),
        "peft_version": package_version("peft"),
        "datasets_version": package_version("datasets"),
        "accelerate_version": package_version("accelerate"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "matplotlib_version": package_version("matplotlib"),
    }

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)

        environment.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_vram_gb": round(
                    properties.total_memory / 1024**3,
                    3,
                ),
                "gpu_compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
            }
        )
    else:
        environment.update(
            {
                "gpu_name": None,
                "gpu_count": 0,
                "gpu_total_vram_gb": None,
                "gpu_compute_capability": None,
            }
        )

    return environment


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def build_training_history(
    log_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge training and validation logs by optimizer step."""

    merged: dict[int, dict[str, Any]] = {}

    for record in log_history:
        step = int(record.get("step", 0))

        row = merged.setdefault(
            step,
            {
                "step": step,
                "epoch": None,
                "train_loss": None,
                "validation_loss": None,
                "learning_rate": None,
                "grad_norm": None,
            },
        )

        if "epoch" in record:
            row["epoch"] = record["epoch"]

        if "loss" in record:
            row["train_loss"] = record["loss"]

        if "eval_loss" in record:
            row["validation_loss"] = record["eval_loss"]

        if "learning_rate" in record:
            row["learning_rate"] = record["learning_rate"]

        if "grad_norm" in record:
            row["grad_norm"] = record["grad_norm"]

    return [
        merged[step]
        for step in sorted(merged)
        if step > 0
    ]


def write_training_history(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "grad_norm",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(
    path: Path,
    history: list[dict[str, Any]],
) -> None:
    train_steps = [
        row["step"]
        for row in history
        if row["train_loss"] is not None
    ]
    train_losses = [
        row["train_loss"]
        for row in history
        if row["train_loss"] is not None
    ]

    validation_steps = [
        row["step"]
        for row in history
        if row["validation_loss"] is not None
    ]
    validation_losses = [
        row["validation_loss"]
        for row in history
        if row["validation_loss"] is not None
    ]

    if not train_losses and not validation_losses:
        LOGGER.warning(
            "Loss curve was not created because no loss values were logged."
        )
        return

    plt.figure(figsize=(9, 6))

    if train_losses:
        plt.plot(
            train_steps,
            train_losses,
            marker="o",
            label="Training loss",
        )

    if validation_losses:
        plt.plot(
            validation_steps,
            validation_losses,
            marker="o",
            label="Validation loss",
        )

    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.title("QLoRA training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def calculate_next_token_accuracy(
    model,
    dataset: Dataset,
    data_collator,
    batch_size: int,
) -> float:
    """Calculate accuracy only for non-masked assistant tokens."""

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    correct_tokens = 0
    evaluated_tokens = 0

    model.eval()

    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                key: value.to(model.device)
                for key, value in batch.items()
            }

            labels = batch["labels"]

            outputs = model(**batch)
            logits = outputs.logits

            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]

            valid_mask = shifted_labels != -100

            predictions = shifted_logits.argmax(dim=-1)

            correct_tokens += (
                (predictions == shifted_labels) & valid_mask
            ).sum().item()

            evaluated_tokens += valid_mask.sum().item()

    if evaluated_tokens == 0:
        return 0.0

    return correct_tokens / evaluated_tokens


def safe_perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()

    artifacts_dir = (
        args.artifacts_dir
        if args.artifacts_dir is not None
        else Path("outputs") / args.language / "training"
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    log_stage(LOGGER, "QLoRA training started")

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    environment = collect_environment()
    write_json(
        artifacts_dir / "environment.json",
        environment,
    )

    log_kv(
        LOGGER,
        {
            "language": args.language,
            "train_jsonl": args.train_jsonl,
            "validation_jsonl": args.validation_jsonl,
            "adapter_dir": args.adapter_dir,
            "artifacts_dir": artifacts_dir,
            "model_name": args.model_name,
            "max_train_samples": args.max_train_samples,
            "max_validation_samples": args.max_validation_samples,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "gpu_name": environment.get("gpu_name"),
            "gpu_total_vram_gb": environment.get(
                "gpu_total_vram_gb"
            ),
        },
    )

    run_config = {
        "language": args.language,
        "base_model": args.model_name,
        "base_model_type": "instruct",
        "base_model_parameter_class": "8B",
        "method": "QLoRA",
        "train_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
        "adapter_dir": str(args.adapter_dir),
        "artifacts_dir": str(artifacts_dir),
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "max_sequence_length": args.max_seq_length,
        "quantization": {
            "load_in_4bit": True,
            "quantization_type": "nf4",
            "compute_dtype": "float16",
            "double_quantization": True,
            "library": "bitsandbytes",
        },
        "lora": {
            "rank": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_batch_size": (
                BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "logging_steps": args.logging_steps,
            "evaluation_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "optimizer": "paged_adamw_8bit",
            "precision": "fp16",
        },
    }

    write_json(
        artifacts_dir / "run_config.json",
        run_config,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. "
            "Run this training script in a GPU environment such as Colab."
        )

    log_stage(LOGGER, "Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        token=hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    log_stage(LOGGER, "Loading base model in 4-bit NF4")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        token=hf_token,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    model.print_trainable_parameters()

    log_stage(LOGGER, "Reading datasets")

    train_rows = read_jsonl(
        args.train_jsonl,
        args.max_train_samples,
    )

    validation_rows = read_jsonl(
        args.validation_jsonl,
        args.max_validation_samples,
    )

    LOGGER.info("Train rows loaded: %s", len(train_rows))
    LOGGER.info(
        "Validation rows loaded: %s",
        len(validation_rows),
    )

    log_stage(LOGGER, "Tokenizing datasets")

    train_dataset = tokenize_rows(
        tokenizer,
        train_rows,
        args.max_seq_length,
    )

    validation_dataset = tokenize_rows(
        tokenizer,
        validation_rows,
        args.max_seq_length,
    )

    LOGGER.info(
        "Tokenized train examples: %s",
        len(train_dataset),
    )
    LOGGER.info(
        "Tokenized validation examples: %s",
        len(validation_dataset),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    checkpoint_dir = (
        args.adapter_dir.parent
        / f"{args.adapter_dir.name}_checkpoints"
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        prediction_loss_only=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    log_stage(LOGGER, "Training")

    training_started_at = time.monotonic()
    train_result = trainer.train()
    training_runtime_seconds = (
        time.monotonic() - training_started_at
    )

    log_stage(LOGGER, "Final validation")

    final_eval_metrics = trainer.evaluate()

    eval_loss = float(
        final_eval_metrics.get("eval_loss", float("nan"))
    )

    validation_perplexity = (
        safe_perplexity(eval_loss)
        if math.isfinite(eval_loss)
        else float("nan")
    )

    log_stage(LOGGER, "Calculating next-token accuracy")

    next_token_accuracy = calculate_next_token_accuracy(
        trainer.model,
        validation_dataset,
        data_collator,
        BATCH_SIZE,
    )

    log_stage(LOGGER, "Saving adapter and tokenizer")

    args.adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.model.save_pretrained(
        args.adapter_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(args.adapter_dir)

    history = build_training_history(
        trainer.state.log_history
    )

    history_path = artifacts_dir / "training_history.csv"
    plot_path = artifacts_dir / "loss_curve.png"

    write_training_history(
        history_path,
        history,
    )

    plot_loss_curve(
        plot_path,
        history,
    )

    trainer.state.save_to_json(
        str(artifacts_dir / "trainer_state.json")
    )

    validation_losses = [
        float(row["validation_loss"])
        for row in history
        if row["validation_loss"] is not None
    ]

    training_metrics = {
        "language": args.language,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "global_steps": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "train_loss": train_result.metrics.get(
            "train_loss"
        ),
        "final_validation_loss": eval_loss,
        "best_validation_loss": (
            min(validation_losses)
            if validation_losses
            else eval_loss
        ),
        "validation_perplexity": validation_perplexity,
        "next_token_accuracy": next_token_accuracy,
        "training_runtime_seconds": (
            training_runtime_seconds
        ),
        "training_runtime_minutes": (
            training_runtime_seconds / 60
        ),
        "total_runtime_seconds": (
            time.monotonic()
            - training_started_at
        ),
        "adapter_path": str(args.adapter_dir),
        "training_history_path": str(history_path),
        "loss_curve_path": str(plot_path),
    }

    write_json(
        artifacts_dir / "training_metrics.json",
        training_metrics,
    )

    log_stage(LOGGER, "Training completed")

    log_kv(
        LOGGER,
        {
            "train_loss": training_metrics["train_loss"],
            "validation_loss": eval_loss,
            "validation_perplexity": validation_perplexity,
            "next_token_accuracy": next_token_accuracy,
            "global_steps": trainer.state.global_step,
            "adapter_dir": args.adapter_dir,
            "training_history": history_path,
            "loss_curve": plot_path,
            "total_elapsed": total_timer.elapsed(),
        },
    )


if __name__ == "__main__":
    main()