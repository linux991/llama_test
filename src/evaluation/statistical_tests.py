"""Statistical comparisons for stylometric feature tables."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def rank_biserial(x_values: list[float], y_values: list[float]) -> float:
    if not x_values or not y_values:
        return float("nan")
    wins = 0.0
    for x_value in x_values:
        for y_value in y_values:
            if x_value > y_value:
                wins += 1.0
            elif x_value == y_value:
                wins += 0.5
    return (2 * wins) / (len(x_values) * len(y_values)) - 1


def mann_whitney_p_value(x_values: list[float], y_values: list[float]) -> float:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x_values, y_values, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")

def wilcoxon_p_value(
    x_values: list[float],
    y_values: list[float],
) -> float:
    """Return a two-sided Wilcoxon signed-rank p-value.

    The two input lists must contain paired measurements in the
    same order. A result below 0.05 indicates a statistically
    significant difference between the paired conditions.
    """

    if len(x_values) != len(y_values):
        raise ValueError(
            "Wilcoxon test requires paired lists "
            "of equal length."
        )

    if not x_values:
        return float("nan")

    differences = [
        x_value - y_value
        for x_value, y_value in zip(
            x_values,
            y_values,
        )
    ]

    # Scipy raises an exception when every paired difference
    # is exactly zero. In that case the conditions are identical.
    if all(
        math.isclose(
            difference,
            0.0,
            abs_tol=1e-12,
        )
        for difference in differences
    ):
        return 1.0

    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            x_values,
            y_values,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return float(result.pvalue)

    except Exception:
        return float("nan")


def paired_rank_biserial(
    x_values: list[float],
    y_values: list[float],
) -> float:
    """Calculate paired rank-biserial effect size.

    Positive values mean that the first condition tends to have
    higher values. Negative values mean that the second condition
    tends to have higher values. Values near zero indicate a weak
    effect, while values near -1 or 1 indicate a strong effect.
    """

    if len(x_values) != len(y_values):
        raise ValueError(
            "Paired effect size requires lists "
            "of equal length."
        )

    differences = [
        x_value - y_value
        for x_value, y_value in zip(
            x_values,
            y_values,
        )
        if not math.isclose(
            x_value - y_value,
            0.0,
            abs_tol=1e-12,
        )
    ]

    if not differences:
        return 0.0

    try:
        from scipy.stats import rankdata

        ranks = rankdata(
            [abs(value) for value in differences],
            method="average",
        )

        positive_rank_sum = sum(
            rank
            for difference, rank in zip(
                differences,
                ranks,
            )
            if difference > 0
        )

        negative_rank_sum = sum(
            rank
            for difference, rank in zip(
                differences,
                ranks,
            )
            if difference < 0
        )

        total_rank_sum = (
            positive_rank_sum
            + negative_rank_sum
        )

        if total_rank_sum == 0:
            return 0.0

        return float(
            (
                positive_rank_sum
                - negative_rank_sum
            )
            / total_rank_sum
        )

    except Exception:
        return float("nan")

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0 if values else float("nan")
    return statistics.stdev(values)


def clean_float(value: str | float | int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")
