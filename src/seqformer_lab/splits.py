"""Leakage-resistant temporal splitting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from seqformer_lab.data import SequenceExample


@dataclass(frozen=True, slots=True)
class TemporalFold:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_periods: tuple[int, ...]
    test_periods: tuple[int, ...]


def rolling_origin_folds(
    examples: Sequence[SequenceExample],
    *,
    minimum_train_periods: int = 2,
    test_periods: int = 1,
    gap_periods: int = 0,
) -> list[TemporalFold]:
    """Create expanding-window folds whose training rows always precede testing rows."""

    if minimum_train_periods <= 0 or test_periods <= 0 or gap_periods < 0:
        raise ValueError("split counts must be positive and gap_periods non-negative")
    periods = sorted({example.period for example in examples})
    needed = minimum_train_periods + gap_periods + test_periods
    if len(periods) < needed:
        raise ValueError(f"need at least {needed} unique periods")

    folds: list[TemporalFold] = []
    first_test = minimum_train_periods + gap_periods
    for test_start in range(first_test, len(periods) - test_periods + 1):
        train_end = test_start - gap_periods
        train_values = tuple(periods[:train_end])
        test_values = tuple(periods[test_start : test_start + test_periods])
        train_set = set(train_values)
        test_set = set(test_values)
        train_indices = tuple(
            i for i, example in enumerate(examples) if example.period in train_set
        )
        test_indices = tuple(i for i, example in enumerate(examples) if example.period in test_set)
        folds.append(
            TemporalFold(
                train_indices=train_indices,
                test_indices=test_indices,
                train_periods=train_values,
                test_periods=test_values,
            )
        )
    return folds


def final_temporal_holdout(
    examples: Sequence[SequenceExample], *, gap_periods: int = 0
) -> TemporalFold:
    """Use the last observed period as a holdout and every eligible earlier period for training."""

    periods = sorted({example.period for example in examples})
    minimum_train = len(periods) - gap_periods - 1
    if minimum_train < 1:
        raise ValueError("not enough periods for the requested holdout gap")
    return rolling_origin_folds(
        examples,
        minimum_train_periods=minimum_train,
        test_periods=1,
        gap_periods=gap_periods,
    )[0]
