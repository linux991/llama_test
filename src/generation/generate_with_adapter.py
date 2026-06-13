#!/usr/bin/env python3
"""Generate reproducible outputs with a trained LoRA adapter.

The script:

1. Loads the same 4-bit base model used for baseline generation.
2. Connects a trained LoRA adapter.
3. Reads prompts only from test_payload.jsonl.
4. Uses the same individual prompts and per-sample seeds as baseline.
5. Saves generation settings and human reference texts to CSV.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from peft import PeftModel

from src.generation.generate_baseline import (
    generate_text,
    load_model,
    read_jsonl,
)
from src.training.lora_config import MODEL_NAME
from src.utils.logging import (
    StepTimer,
    format_seconds,
    get_logger,
    log_kv,
    log_stage,
)


LOGGER = get_logger("generate_with_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        choices=["en", "ru"],
        required=True,
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base seed. The seed of sample N is "
            "base_seed + N - 1."
        ),
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token. Defaults to the "
            "HF_TOKEN environment variable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()

    if not args.adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter directory not found: "
            f"{args.adapter_dir}"
        )

    log_stage(
        LOGGER,
        "Fine-tuned generation started",
    )

    log_kv(
        LOGGER,
        {
            "language": args.language,
            "test_jsonl": args.test_jsonl,
            "adapter_dir": args.adapter_dir,
            "output_csv": args.output_csv,
            "model_name": args.model_name,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": (
                args.repetition_penalty
            ),
            "base_seed": args.seed,
            "load_in_4bit": args.load_in_4bit,
        },
    )

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)

        log_kv(
            LOGGER,
            {
                "cuda_available": True,
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                ),
                "cuda_total_vram_gb": round(
                    properties.total_memory / 1024**3,
                    3,
                ),
            },
        )
    else:
        LOGGER.warning(
            "CUDA is not available. "
            "This script is intended for a GPU runtime."
        )

    log_stage(
        LOGGER,
        "Reading held-out test payload",
    )

    payload_rows = read_jsonl(
        args.test_jsonl,
        args.max_samples,
    )

    LOGGER.info(
        "Loaded %s test rows from %s.",
        len(payload_rows),
        args.test_jsonl,
    )

    tokenizer, base_model = load_model(
        args.model_name,
        args.load_in_4bit,
        args.hf_token,
    )

    log_stage(
        LOGGER,
        "Loading LoRA adapter",
    )

    adapter_timer = StepTimer()

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        is_trainable=False,
    )

    model.config.use_cache = True
    model.eval()

    LOGGER.info(
        "Adapter loaded from %s in %s.",
        args.adapter_dir,
        adapter_timer.elapsed(),
    )

    log_stage(
        LOGGER,
        "Generating fine-tuned outputs",
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "story_id",
        "language",
        "condition",
        "system_message",
        "prompt",
        "reference_text",
        "generated_text",
        "source",
        "title",
        "model_name",
        "adapter_name",
        "generation_temperature",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
        "do_sample",
        "base_seed",
        "sample_seed",
        "load_in_4bit",
        "generation_seconds",
    ]

    with args.output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index, row in enumerate(
            payload_rows,
            start=1,
        ):
            row_started_at = time.monotonic()

            prompt = str(row["prompt"]).strip()
            system_message = str(
                row.get("system", "")
            ).strip()

            # Формула полностью совпадает с baseline.
            sample_seed = args.seed + index - 1

            generated_text = generate_text(
                tokenizer,
                model,
                system_message,
                prompt,
                args,
                sample_seed,
            )

            generation_seconds = (
                time.monotonic() - row_started_at
            )

            writer.writerow(
                {
                    "id": row["id"],
                    "story_id": row.get(
                        "story_id",
                        "",
                    ),
                    "language": args.language,
                    "condition": "finetuned",
                    "system_message": system_message,
                    "prompt": prompt,
                    "reference_text": row.get(
                        "text",
                        "",
                    ),
                    "generated_text": generated_text,
                    "source": row.get(
                        "source",
                        "",
                    ),
                    "title": row.get(
                        "title",
                        "",
                    ),
                    "model_name": args.model_name,
                    "adapter_name": str(
                        args.adapter_dir
                    ),
                    "generation_temperature": (
                        args.temperature
                    ),
                    "top_p": args.top_p,
                    "repetition_penalty": (
                        args.repetition_penalty
                    ),
                    "max_new_tokens": (
                        args.max_new_tokens
                    ),
                    "do_sample": True,
                    "base_seed": args.seed,
                    "sample_seed": sample_seed,
                    "load_in_4bit": (
                        args.load_in_4bit
                    ),
                    "generation_seconds": round(
                        generation_seconds,
                        4,
                    ),
                }
            )

            # Сохраняем каждую строку сразу.
            # Если Colab отключится, уже полученные
            # генерации останутся в CSV.
            handle.flush()

            if (
                index == 1
                or index % 5 == 0
                or index == len(payload_rows)
            ):
                LOGGER.info(
                    "Generated %s/%s rows | "
                    "sample_seed=%s | "
                    "last_row=%s | total_elapsed=%s",
                    index,
                    len(payload_rows),
                    sample_seed,
                    format_seconds(
                        generation_seconds
                    ),
                    total_timer.elapsed(),
                )

    log_stage(
        LOGGER,
        "Fine-tuned generation finished",
    )

    LOGGER.info(
        "Saved CSV: %s",
        args.output_csv,
    )
    LOGGER.info(
        "Total elapsed: %s",
        total_timer.elapsed(),
    )


if __name__ == "__main__":
    main()