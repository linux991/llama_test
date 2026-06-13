#!/usr/bin/env python3
"""Validate paired generation outputs and calculate stylometric statistics.

The script:

1. Reads human reference texts from the held-out test JSONL.
2. Reads baseline and fine-tuned generation CSV files.
3. Verifies that IDs, prompts, reference texts, seeds and generation
   settings match between paired conditions.
4. Extracts stylometric features for human, baseline and fine-tuned texts.
5. Applies paired Wilcoxon signed-rank tests.
6. Calculates paired rank-biserial effect sizes.
7. Saves a machine-readable validation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    FEATURE_COLUMNS,
    stylometric_features,
)
from src.evaluation.statistical_tests import (
    clean_float,
    mean,
    median,
    paired_rank_biserial,
    stdev,
    wilcoxon_p_value,
)
from src.utils.logging import (
    StepTimer,
    get_logger,
    log_kv,
    log_stage,
)


LOGGER = get_logger("run_stylometry")


GENERATION_SETTING_COLUMNS = [
    "model_name",
    "generation_temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "do_sample",
    "base_seed",
    "load_in_4bit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        choices=["en", "ru"],
        required=True,
    )

    # --human-jsonl remains as a backward-compatible alias.
    parser.add_argument(
        "--test-jsonl",
        "--human-jsonl",
        dest="test_jsonl",
        type=Path,
        required=True,
        help=(
            "Held-out test_payload.jsonl containing human "
            "reference texts."
        ),
    )

    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--finetuned-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help=(
            "Optional JSON report path. Defaults to "
            "comparison_validation.json next to summary CSV."
        ),
    )

    return parser.parse_args()


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def read_test_jsonl(
    path: Path,
    language: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test JSONL file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            item = json.loads(line)

            row_id = str(
                item.get("id", "")
            ).strip()

            prompt = str(
                item.get("prompt", "")
            ).strip()

            text = str(
                item.get("text", "")
            ).strip()

            row_language = str(
                item.get("language", language)
            ).strip()

            split = str(
                item.get("split", "")
            ).strip()

            if not row_id:
                raise ValueError(
                    f"Missing ID in {path}, "
                    f"line {line_number}."
                )

            if not prompt:
                raise ValueError(
                    f"Empty prompt in {path}, "
                    f"line {line_number}."
                )

            if not text:
                raise ValueError(
                    f"Empty human text in {path}, "
                    f"line {line_number}."
                )

            if row_language != language:
                raise ValueError(
                    f"Language mismatch in {path}, "
                    f"line {line_number}: "
                    f"expected {language!r}, "
                    f"received {row_language!r}."
                )

            if split and split != "test":
                raise ValueError(
                    f"Expected split='test' in {path}, "
                    f"line {line_number}, "
                    f"received {split!r}."
                )

            rows.append(
                {
                    "id": row_id,
                    "story_id": item.get(
                        "story_id",
                        "",
                    ),
                    "language": language,
                    "condition": "human",
                    "system_message": str(
                        item.get("system", "")
                    ).strip(),
                    "prompt": prompt,
                    "reference_text": text,
                    "text": text,
                    "sample_seed": "",
                    "source": item.get(
                        "source",
                        "",
                    ),
                    "title": item.get(
                        "title",
                        "",
                    ),
                }
            )

    if not rows:
        raise ValueError(
            f"No usable test rows found in {path}."
        )

    return rows


def read_generation_csv(
    path: Path,
    expected_condition: str,
    language: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generation CSV file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {path}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            row_id = str(
                row.get("id", "")
            ).strip()

            prompt = str(
                row.get("prompt", "")
            ).strip()

            generated_text = str(
                row.get("generated_text", "")
            ).strip()

            condition = str(
                row.get(
                    "condition",
                    expected_condition,
                )
            ).strip()

            row_language = str(
                row.get(
                    "language",
                    language,
                )
            ).strip()

            if not row_id:
                raise ValueError(
                    f"Missing ID in {path}, "
                    f"line {line_number}."
                )

            if not prompt:
                raise ValueError(
                    f"Empty prompt in {path}, "
                    f"line {line_number}."
                )

            if not generated_text:
                raise ValueError(
                    f"Empty generated text in {path}, "
                    f"line {line_number}."
                )

            if condition != expected_condition:
                raise ValueError(
                    f"Condition mismatch in {path}, "
                    f"line {line_number}: expected "
                    f"{expected_condition!r}, "
                    f"received {condition!r}."
                )

            if row_language != language:
                raise ValueError(
                    f"Language mismatch in {path}, "
                    f"line {line_number}: expected "
                    f"{language!r}, "
                    f"received {row_language!r}."
                )

            normalized_row: dict[str, Any] = {
                **row,
                "id": row_id,
                "story_id": row.get(
                    "story_id",
                    "",
                ),
                "language": language,
                "condition": expected_condition,
                "system_message": str(
                    row.get(
                        "system_message",
                        "",
                    )
                ).strip(),
                "prompt": prompt,
                "reference_text": str(
                    row.get(
                        "reference_text",
                        "",
                    )
                ).strip(),
                "text": generated_text,
                "sample_seed": str(
                    row.get(
                        "sample_seed",
                        "",
                    )
                ).strip(),
            }

            rows.append(normalized_row)

    if not rows:
        raise ValueError(
            f"No generation rows found in {path}."
        )

    return rows


def index_unique_rows(
    rows: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for row in rows:
        row_id = str(row["id"])

        if row_id in indexed:
            raise ValueError(
                f"Duplicate ID {row_id!r} "
                f"in {source_name}."
            )

        indexed[row_id] = row

    return indexed


def normalized_setting(
    value: Any,
) -> str:
    return str(value).strip().lower()


def validate_paired_inputs(
    test_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
    report_path: Path,
    language: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    test_by_id = index_unique_rows(
        test_rows,
        "test JSONL",
    )

    baseline_by_id = index_unique_rows(
        baseline_rows,
        "baseline CSV",
    )

    finetuned_by_id = index_unique_rows(
        finetuned_rows,
        "fine-tuned CSV",
    )

    test_ids = set(test_by_id)
    baseline_ids = set(baseline_by_id)
    finetuned_ids = set(finetuned_by_id)

    errors: list[str] = []

    missing_in_baseline = sorted(
        test_ids - baseline_ids
    )

    missing_in_finetuned = sorted(
        test_ids - finetuned_ids
    )

    unexpected_baseline = sorted(
        baseline_ids - test_ids
    )

    unexpected_finetuned = sorted(
        finetuned_ids - test_ids
    )

    if missing_in_baseline:
        errors.append(
            "IDs missing from baseline: "
            + ", ".join(missing_in_baseline[:10])
        )

    if missing_in_finetuned:
        errors.append(
            "IDs missing from fine-tuned: "
            + ", ".join(missing_in_finetuned[:10])
        )

    if unexpected_baseline:
        errors.append(
            "Unexpected baseline IDs: "
            + ", ".join(unexpected_baseline[:10])
        )

    if unexpected_finetuned:
        errors.append(
            "Unexpected fine-tuned IDs: "
            + ", ".join(
                unexpected_finetuned[:10]
            )
        )

    common_ids = sorted(
        test_ids
        & baseline_ids
        & finetuned_ids
    )

    prompt_mismatches: list[str] = []
    system_mismatches: list[str] = []
    reference_mismatches: list[str] = []
    seed_mismatches: list[str] = []
    setting_mismatches: list[str] = []

    for row_id in common_ids:
        human_row = test_by_id[row_id]
        baseline_row = baseline_by_id[row_id]
        finetuned_row = finetuned_by_id[row_id]

        human_prompt = str(
            human_row["prompt"]
        ).strip()

        baseline_prompt = str(
            baseline_row["prompt"]
        ).strip()

        finetuned_prompt = str(
            finetuned_row["prompt"]
        ).strip()

        if not (
            human_prompt
            == baseline_prompt
            == finetuned_prompt
        ):
            prompt_mismatches.append(row_id)

        human_system = str(
            human_row.get(
                "system_message",
                "",
            )
        ).strip()

        baseline_system = str(
            baseline_row.get(
                "system_message",
                "",
            )
        ).strip()

        finetuned_system = str(
            finetuned_row.get(
                "system_message",
                "",
            )
        ).strip()

        if not (
            human_system
            == baseline_system
            == finetuned_system
        ):
            system_mismatches.append(row_id)

        human_reference = str(
            human_row["reference_text"]
        ).strip()

        baseline_reference = str(
            baseline_row.get(
                "reference_text",
                "",
            )
        ).strip()

        finetuned_reference = str(
            finetuned_row.get(
                "reference_text",
                "",
            )
        ).strip()

        if baseline_reference and (
            baseline_reference
            != human_reference
        ):
            reference_mismatches.append(
                f"{row_id}: baseline"
            )

        if finetuned_reference and (
            finetuned_reference
            != human_reference
        ):
            reference_mismatches.append(
                f"{row_id}: finetuned"
            )

        baseline_seed = str(
            baseline_row.get(
                "sample_seed",
                "",
            )
        ).strip()

        finetuned_seed = str(
            finetuned_row.get(
                "sample_seed",
                "",
            )
        ).strip()

        if (
            not baseline_seed
            or not finetuned_seed
            or baseline_seed != finetuned_seed
        ):
            seed_mismatches.append(row_id)

        for setting_name in (
            GENERATION_SETTING_COLUMNS
        ):
            baseline_value = normalized_setting(
                baseline_row.get(
                    setting_name,
                    "",
                )
            )

            finetuned_value = normalized_setting(
                finetuned_row.get(
                    setting_name,
                    "",
                )
            )

            if baseline_value != finetuned_value:
                setting_mismatches.append(
                    f"{row_id}: {setting_name}"
                )

    if prompt_mismatches:
        errors.append(
            "Prompt mismatches: "
            + ", ".join(
                prompt_mismatches[:10]
            )
        )

    if system_mismatches:
        errors.append(
            "System-message mismatches: "
            + ", ".join(
                system_mismatches[:10]
            )
        )

    if reference_mismatches:
        errors.append(
            "Reference-text mismatches: "
            + ", ".join(
                reference_mismatches[:10]
            )
        )

    if seed_mismatches:
        errors.append(
            "Sample-seed mismatches: "
            + ", ".join(
                seed_mismatches[:10]
            )
        )

    if setting_mismatches:
        errors.append(
            "Generation-setting mismatches: "
            + ", ".join(
                setting_mismatches[:10]
            )
        )

    report = {
        "language": language,
        "test_row_count": len(test_rows),
        "baseline_row_count": len(
            baseline_rows
        ),
        "finetuned_row_count": len(
            finetuned_rows
        ),
        "common_id_count": len(common_ids),
        "missing_in_baseline_count": len(
            missing_in_baseline
        ),
        "missing_in_finetuned_count": len(
            missing_in_finetuned
        ),
        "unexpected_baseline_count": len(
            unexpected_baseline
        ),
        "unexpected_finetuned_count": len(
            unexpected_finetuned
        ),
        "prompt_mismatch_count": len(
            prompt_mismatches
        ),
        "system_message_mismatch_count": len(
            system_mismatches
        ),
        "reference_text_mismatch_count": len(
            reference_mismatches
        ),
        "sample_seed_mismatch_count": len(
            seed_mismatches
        ),
        "generation_setting_mismatch_count": len(
            setting_mismatches
        ),
        "paired_comparison_valid": not errors,
        "errors": errors,
    }

    write_json(
        report_path,
        report,
    )

    if errors:
        preview = "\n".join(
            f"- {error}"
            for error in errors[:10]
        )

        raise RuntimeError(
            "Paired comparison validation failed. "
            f"See {report_path}.\n{preview}"
        )

    return (
        test_by_id,
        baseline_by_id,
        finetuned_by_id,
    )


def build_aligned_rows(
    test_by_id: dict[str, dict[str, Any]],
    baseline_by_id: dict[
        str,
        dict[str, Any],
    ],
    finetuned_by_id: dict[
        str,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    aligned_rows: list[dict[str, Any]] = []

    for row_id in sorted(test_by_id):
        aligned_rows.extend(
            [
                test_by_id[row_id],
                baseline_by_id[row_id],
                finetuned_by_id[row_id],
            ]
        )

    return aligned_rows


def write_features(
    path: Path,
    source_rows: list[dict[str, Any]],
    language: str,
) -> list[dict[str, Any]]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for row in source_rows:
        features = stylometric_features(
            str(row["text"]),
            language,
        )

        rows.append(
            {
                **row,
                **features,
            }
        )

    fieldnames = [
        "id",
        "story_id",
        "language",
        "condition",
        "prompt",
        "sample_seed",
        "text",
        *FEATURE_COLUMNS,
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def effect_size_magnitude(
    value: float,
) -> str:
    if not math.isfinite(value):
        return "undefined"

    absolute_value = abs(value)

    if absolute_value < 0.1:
        return "negligible"

    if absolute_value < 0.3:
        return "small"

    if absolute_value < 0.5:
        return "medium"

    return "large"


def paired_feature_values(
    feature_rows: list[dict[str, Any]],
    left_condition: str,
    right_condition: str,
    feature: str,
) -> tuple[
    list[float],
    list[float],
]:
    rows_by_condition: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for condition in (
        left_condition,
        right_condition,
    ):
        rows_by_condition[condition] = {
            str(row["id"]): row
            for row in feature_rows
            if row["condition"] == condition
        }

    common_ids = sorted(
        set(
            rows_by_condition[
                left_condition
            ]
        )
        & set(
            rows_by_condition[
                right_condition
            ]
        )
    )

    left_values: list[float] = []
    right_values: list[float] = []

    for row_id in common_ids:
        left_value = clean_float(
            rows_by_condition[
                left_condition
            ][row_id][feature]
        )

        right_value = clean_float(
            rows_by_condition[
                right_condition
            ][row_id][feature]
        )

        if (
            math.isfinite(left_value)
            and math.isfinite(right_value)
        ):
            left_values.append(left_value)
            right_values.append(right_value)

    return left_values, right_values


def write_summary(
    path: Path,
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = [
        ("human", "baseline"),
        ("human", "finetuned"),
        ("baseline", "finetuned"),
    ]

    summary_rows: list[dict[str, Any]] = []

    for left, right in comparisons:
        for feature in FEATURE_COLUMNS:
            left_values, right_values = (
                paired_feature_values(
                    feature_rows,
                    left,
                    right,
                    feature,
                )
            )

            if not left_values:
                continue

            differences = [
                left_value - right_value
                for left_value, right_value in zip(
                    left_values,
                    right_values,
                )
            ]

            p_value = wilcoxon_p_value(
                left_values,
                right_values,
            )

            effect_size = paired_rank_biserial(
                left_values,
                right_values,
            )

            summary_rows.append(
                {
                    "comparison": (
                        f"{left}_vs_{right}"
                    ),
                    "feature": feature,
                    "left_condition": left,
                    "right_condition": right,
                    "n_pairs": len(left_values),
                    "left_mean": mean(
                        left_values
                    ),
                    "right_mean": mean(
                        right_values
                    ),
                    "mean_difference_left_minus_right": (
                        mean(differences)
                    ),
                    "left_median": median(
                        left_values
                    ),
                    "right_median": median(
                        right_values
                    ),
                    "median_paired_difference": median(
                        differences
                    ),
                    "left_std": stdev(
                        left_values
                    ),
                    "right_std": stdev(
                        right_values
                    ),
                    "wilcoxon_p_value": p_value,
                    "significant_at_0_05": (
                        math.isfinite(p_value)
                        and p_value < 0.05
                    ),
                    "paired_rank_biserial": (
                        effect_size
                    ),
                    "effect_size_magnitude": (
                        effect_size_magnitude(
                            effect_size
                        )
                    ),
                }
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "comparison",
        "feature",
        "left_condition",
        "right_condition",
        "n_pairs",
        "left_mean",
        "right_mean",
        "mean_difference_left_minus_right",
        "left_median",
        "right_median",
        "median_paired_difference",
        "left_std",
        "right_std",
        "wilcoxon_p_value",
        "significant_at_0_05",
        "paired_rank_biserial",
        "effect_size_magnitude",
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
        writer.writerows(summary_rows)

    return summary_rows


def main() -> None:
    timer = StepTimer()
    args = parse_args()

    validation_report = (
        args.validation_report
        if args.validation_report is not None
        else args.summary_csv.with_name(
            "comparison_validation.json"
        )
    )

    log_stage(
        LOGGER,
        "Paired stylometry started",
    )

    log_kv(
        LOGGER,
        {
            "language": args.language,
            "test_jsonl": args.test_jsonl,
            "baseline_csv": args.baseline_csv,
            "finetuned_csv": (
                args.finetuned_csv
            ),
            "features_csv": args.features_csv,
            "summary_csv": args.summary_csv,
            "validation_report": (
                validation_report
            ),
        },
    )

    test_rows = read_test_jsonl(
        args.test_jsonl,
        args.language,
    )

    baseline_rows = read_generation_csv(
        args.baseline_csv,
        "baseline",
        args.language,
    )

    finetuned_rows = read_generation_csv(
        args.finetuned_csv,
        "finetuned",
        args.language,
    )

    log_stage(
        LOGGER,
        "Validating paired inputs",
    )

    (
        test_by_id,
        baseline_by_id,
        finetuned_by_id,
    ) = validate_paired_inputs(
        test_rows,
        baseline_rows,
        finetuned_rows,
        validation_report,
        args.language,
    )

    LOGGER.info(
        "Paired validation passed for "
        "%s test examples.",
        len(test_by_id),
    )

    source_rows = build_aligned_rows(
        test_by_id,
        baseline_by_id,
        finetuned_by_id,
    )

    log_stage(
        LOGGER,
        "Extracting stylometric features",
    )

    feature_rows = write_features(
        args.features_csv,
        source_rows,
        args.language,
    )

    log_stage(
        LOGGER,
        "Calculating paired statistics",
    )

    summary_rows = write_summary(
        args.summary_csv,
        feature_rows,
    )

    log_stage(
        LOGGER,
        "Paired stylometry finished",
    )

    LOGGER.info(
        "Wrote %s feature rows to %s.",
        len(feature_rows),
        args.features_csv,
    )

    LOGGER.info(
        "Wrote %s statistical rows to %s.",
        len(summary_rows),
        args.summary_csv,
    )

    LOGGER.info(
        "Validation report: %s.",
        validation_report,
    )

    LOGGER.info(
        "Total elapsed: %s.",
        timer.elapsed(),
    )


if __name__ == "__main__":
    main()