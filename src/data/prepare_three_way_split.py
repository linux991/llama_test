#!/usr/bin/env python3
"""Create deterministic train/validation/test JSONL splits for EN and RU.

The source data already contain a story-level train/eval separation.
This script:

1. Takes training examples from the existing train CSV.
2. Divides the existing holdout/eval CSV into validation and test sets.
3. Keeps story IDs separate between all three sets.
4. Adds system, user and assistant fields for instruction tuning.
5. Writes metadata and verifies that no story leakage exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SETTINGS = {
    "en": {
        "train_csv": Path(
            "archive/old_experiments/payloads/en_csv/"
            "train_with_descriptions.csv"
        ),
        "holdout_csv": Path(
            "archive/old_experiments/payloads/en_csv/"
            "eval_with_descriptions.csv"
        ),
        "output_dir": Path("data/processed/en"),
        "source": "english_creepypasta",
        "system_message": (
            "You are a literary fiction writer specializing in horror "
            "and thriller. Write only the requested fictional scene "
            "without explanations or commentary."
        ),
    },
    "ru": {
        "train_csv": Path(
            "archive/old_experiments/payloads/ru_csv/"
            "train_with_descriptions.csv"
        ),
        "holdout_csv": Path(
            "archive/old_experiments/payloads/ru_csv/"
            "eval_with_descriptions.csv"
        ),
        "output_dir": Path("data/processed/ru"),
        "source": "russian_creepypasta",
        "system_message": (
            "Ты автор художественной прозы, специализирующийся "
            "на ужасах и триллерах. Пиши только запрошенную сцену "
            "без пояснений и метакомментариев."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        choices=["en", "ru", "all"],
        default="all",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Source CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Source CSV is empty: {path}")

    return rows


def get_story_id(row: dict[str, str], row_number: int) -> str:
    """Return a stable story identifier.

    story_id is preferred. source_file and title are used as fallbacks.
    A row-specific fallback is used only when no identifying field exists.
    """

    candidates = [
        row.get("story_id", ""),
        row.get("source_file", ""),
        row.get("title", ""),
    ]

    for value in candidates:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned

    return f"unknown_story_{row_number:08d}"


def add_resolved_story_ids(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    resolved_rows = []

    for index, row in enumerate(rows, start=1):
        copied_row = dict(row)
        copied_row["_resolved_story_id"] = get_story_id(row, index)
        resolved_rows.append(copied_row)

    return resolved_rows


def group_by_story(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        groups[row["_resolved_story_id"]].append(row)

    return dict(groups)


def select_training_rows(
    rows: list[dict[str, str]],
    target_size: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select training examples deterministically by shuffled story groups."""

    groups = group_by_story(rows)
    story_ids = sorted(groups)

    rng = random.Random(seed)
    rng.shuffle(story_ids)

    selected: list[dict[str, str]] = []

    for story_id in story_ids:
        if len(selected) >= target_size:
            break

        remaining = target_size - len(selected)
        selected.extend(groups[story_id][:remaining])

    if len(selected) < target_size:
        raise ValueError(
            f"Not enough training rows: requested {target_size}, "
            f"found {len(selected)}."
        )

    return selected


def split_holdout_rows(
    rows: list[dict[str, str]],
    validation_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split holdout data by whole story IDs.

    A story assigned to validation can never appear in test.
    Unused chunks from a selected story are discarded rather than
    moved into another split.
    """

    groups = group_by_story(rows)
    story_ids = sorted(groups)

    rng = random.Random(seed)
    rng.shuffle(story_ids)

    validation_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    validation_story_ids: set[str] = set()
    test_story_ids: set[str] = set()

    for story_id in story_ids:
        story_rows = groups[story_id]

        if len(validation_rows) < validation_size:
            remaining = validation_size - len(validation_rows)
            validation_rows.extend(story_rows[:remaining])
            validation_story_ids.add(story_id)
            continue

        if len(test_rows) < test_size:
            remaining = test_size - len(test_rows)
            test_rows.extend(story_rows[:remaining])
            test_story_ids.add(story_id)

        if (
            len(validation_rows) >= validation_size
            and len(test_rows) >= test_size
        ):
            break

    if len(validation_rows) < validation_size:
        raise ValueError(
            f"Not enough validation rows: requested {validation_size}, "
            f"found {len(validation_rows)}."
        )

    if len(test_rows) < test_size:
        raise ValueError(
            f"Not enough test rows: requested {test_size}, "
            f"found {len(test_rows)}."
        )

    overlap = validation_story_ids & test_story_ids

    if overlap:
        raise RuntimeError(
            "Story leakage between validation and test: "
            f"{sorted(overlap)}"
        )

    return validation_rows, test_rows


def stable_id(
    language: str,
    split: str,
    index: int,
    row: dict[str, str],
) -> str:
    raw = "|".join(
        [
            language,
            split,
            str(index),
            row["_resolved_story_id"],
            row.get("chunk_id", ""),
            row.get("text_sha256", ""),
        ]
    )

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]

    return f"{language}_{split}_{index:06d}_{digest}"


def normalize_row(
    row: dict[str, str],
    *,
    language: str,
    split: str,
    index: int,
    source: str,
    system_message: str,
) -> dict[str, Any]:
    chunk_id_raw = row.get("chunk_id", "")

    try:
        chunk_id = int(chunk_id_raw or 0)
    except ValueError:
        chunk_id = 0

    return {
        "id": stable_id(language, split, index, row),
        "story_id": row["_resolved_story_id"],
        "language": language,
        "genre": "horror_thriller",
        "system": system_message,
        "prompt": row.get("description", "").strip(),
        "text": row.get("text", "").strip(),
        "source": source,
        "split": split,
        "source_file": row.get("source_file", ""),
        "title": row.get("title", ""),
        "chunk_id": chunk_id,
        "text_sha256": row.get("text_sha256", ""),
    }


def normalize_rows(
    rows: list[dict[str, str]],
    *,
    language: str,
    split: str,
    source: str,
    system_message: str,
) -> list[dict[str, Any]]:
    return [
        normalize_row(
            row,
            language=language,
            split=split,
            index=index,
            source=source,
            system_message=system_message,
        )
        for index, row in enumerate(rows, start=1)
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def story_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["story_id"]) for row in rows}


def verify_no_leakage(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, int]:
    train_ids = story_ids(train_rows)
    validation_ids = story_ids(validation_rows)
    test_ids = story_ids(test_rows)

    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids

    if train_validation_overlap:
        raise RuntimeError(
            "Leakage between train and validation: "
            f"{sorted(train_validation_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "Leakage between train and test: "
            f"{sorted(train_test_overlap)}"
        )

    if validation_test_overlap:
        raise RuntimeError(
            "Leakage between validation and test: "
            f"{sorted(validation_test_overlap)}"
        )

    return {
        "train_validation_story_overlap": 0,
        "train_test_story_overlap": 0,
        "validation_test_story_overlap": 0,
    }


def prepare_language(
    language: str,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, Any]:
    settings = SETTINGS[language]

    raw_train_rows = add_resolved_story_ids(
        read_csv(settings["train_csv"])
    )
    raw_holdout_rows = add_resolved_story_ids(
        read_csv(settings["holdout_csv"])
    )

    source_train_story_ids = {
        row["_resolved_story_id"] for row in raw_train_rows
    }
    source_holdout_story_ids = {
        row["_resolved_story_id"] for row in raw_holdout_rows
    }

    source_overlap = (
        source_train_story_ids & source_holdout_story_ids
    )

    if source_overlap:
        raise RuntimeError(
            "The original train and holdout CSV files already contain "
            f"overlapping stories: {sorted(source_overlap)}"
        )

    selected_train = select_training_rows(
        raw_train_rows,
        target_size=train_size,
        seed=seed,
    )

    selected_validation, selected_test = split_holdout_rows(
        raw_holdout_rows,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    train_rows = normalize_rows(
        selected_train,
        language=language,
        split="train",
        source=settings["source"],
        system_message=settings["system_message"],
    )

    validation_rows = normalize_rows(
        selected_validation,
        language=language,
        split="validation",
        source=settings["source"],
        system_message=settings["system_message"],
    )

    test_rows = normalize_rows(
        selected_test,
        language=language,
        split="test",
        source=settings["source"],
        system_message=settings["system_message"],
    )

    leakage_report = verify_no_leakage(
        train_rows,
        validation_rows,
        test_rows,
    )

    output_dir: Path = settings["output_dir"]

    train_path = output_dir / "train_payload.jsonl"
    validation_path = output_dir / "validation_payload.jsonl"
    test_path = output_dir / "test_payload.jsonl"
    metadata_path = output_dir / "metadata.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(test_path, test_rows)

    metadata = {
        "language": language,
        "seed": seed,
        "source_train_csv": str(settings["train_csv"]),
        "source_holdout_csv": str(settings["holdout_csv"]),
        "source_train_rows": len(raw_train_rows),
        "source_holdout_rows": len(raw_holdout_rows),
        "written_train_rows": len(train_rows),
        "written_validation_rows": len(validation_rows),
        "written_test_rows": len(test_rows),
        "train_story_count": len(story_ids(train_rows)),
        "validation_story_count": len(story_ids(validation_rows)),
        "test_story_count": len(story_ids(test_rows)),
        "story_level_split": True,
        **leakage_report,
        "train_jsonl": str(train_path),
        "validation_jsonl": str(validation_path),
        "test_jsonl": str(test_path),
        "notes": (
            "Train originates from the existing story-level train pool. "
            "Validation and test originate from separate story groups "
            "inside the existing holdout pool."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Language: {language}")
    print(f"Train rows:      {len(train_rows)}")
    print(f"Validation rows: {len(validation_rows)}")
    print(f"Test rows:       {len(test_rows)}")
    print(f"Train stories:      {len(story_ids(train_rows))}")
    print(f"Validation stories: {len(story_ids(validation_rows))}")
    print(f"Test stories:       {len(story_ids(test_rows))}")
    print("Story overlap: 0")
    print(f"Metadata: {metadata_path}")

    return metadata


def main() -> None:
    args = parse_args()

    languages = (
        ["en", "ru"]
        if args.language == "all"
        else [args.language]
    )

    summaries = []

    for language in languages:
        summary = prepare_language(
            language,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        summaries.append(summary)

    print()
    print("All requested data splits were created successfully.")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()