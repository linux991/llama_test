#!/usr/bin/env python3
"""Generate reproducible baseline outputs on the held-out test split.

The script:

1. Loads the base Llama 3 model without a LoRA adapter.
2. Reads prompts only from test_payload.jsonl.
3. Uses the system message and individual prompt from each test row.
4. Assigns a deterministic seed to every sample.
5. Saves generation settings and the human reference text to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.training.lora_config import MODEL_NAME
from src.utils.logging import (
    StepTimer,
    format_seconds,
    get_logger,
    log_kv,
    log_stage,
)


LOGGER = get_logger("generate_baseline")


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


def read_jsonl(
    path: Path,
    limit: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test JSONL file not found: {path}"
        )

    if limit <= 0:
        raise ValueError(
            "--max-samples must be greater than zero."
        )

    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            row_id = str(row.get("id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            split = str(row.get("split", "")).strip()

            if not row_id:
                raise ValueError(
                    f"Missing id in {path}, line {line_number}."
                )

            if not prompt:
                raise ValueError(
                    f"Empty prompt in {path}, line {line_number}."
                )

            if split and split != "test":
                raise ValueError(
                    f"Expected split='test' in {path}, "
                    f"line {line_number}, received {split!r}."
                )

            rows.append(row)

            if len(rows) >= limit:
                break

    if not rows:
        raise ValueError(
            f"No usable test rows found in {path}."
        )

    return rows


def resolve_hf_token(
    token: str | None = None,
) -> str | None:
    return token or os.environ.get("HF_TOKEN")


def load_model(
    model_name: str,
    load_in_4bit: bool,
    hf_token: str | None = None,
):
    token = resolve_hf_token(hf_token)

    log_stage(LOGGER, "Loading tokenizer")

    log_kv(
        LOGGER,
        {
            "model_name": model_name,
            "hf_token_present": bool(token),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    quantization_config = None

    if load_in_4bit:
        LOGGER.info(
            "Using 4-bit NF4 quantization with "
            "float16 compute and double quantization."
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    log_stage(LOGGER, "Loading base model")

    timer = StepTimer()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        token=token,
    )

    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    LOGGER.info(
        "Model ready in %s.",
        timer.elapsed(),
    )

    return tokenizer, model


def build_chat_prompt(
    tokenizer,
    system_message: str,
    prompt: str,
) -> str:
    messages: list[dict[str, str]] = []

    if system_message.strip():
        messages.append(
            {
                "role": "system",
                "content": system_message.strip(),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt.strip(),
        }
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_text(
    tokenizer,
    model,
    system_message: str,
    prompt: str,
    args: argparse.Namespace,
    sample_seed: int,
) -> str:
    # Каждый тестовый пример получает собственный seed.
    # В скрипте fine-tuned генерации будет использована
    # точно такая же формула.
    set_seed(sample_seed)

    chat_prompt = build_chat_prompt(
        tokenizer,
        system_message,
        prompt,
    )

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[-1] :,
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def main() -> None:
    total_timer = StepTimer()
    args = parse_args()

    log_stage(
        LOGGER,
        "Baseline generation started",
    )

    log_kv(
        LOGGER,
        {
            "language": args.language,
            "test_jsonl": args.test_jsonl,
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

    set_seed(args.seed)

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

    tokenizer, model = load_model(
        args.model_name,
        args.load_in_4bit,
        args.hf_token,
    )

    log_stage(
        LOGGER,
        "Generating baseline outputs",
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
                    "condition": "baseline",
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
                    "adapter_name": "",
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
        "Baseline generation finished",
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