import pytest

from seqformer_lab.data import generate_synthetic_examples
from seqformer_lab.splits import final_temporal_holdout, rolling_origin_folds


def test_folds_never_train_on_future_periods() -> None:
    examples = generate_synthetic_examples(periods=6, examples_per_period=3)
    folds = rolling_origin_folds(examples, minimum_train_periods=2, gap_periods=1)
    assert len(folds) == 3
    for fold in folds:
        assert max(fold.train_periods) < min(fold.test_periods)
        assert set(fold.train_indices).isdisjoint(fold.test_indices)


def test_final_holdout_uses_last_period() -> None:
    examples = generate_synthetic_examples(periods=4, examples_per_period=2)
    fold = final_temporal_holdout(examples)
    assert fold.train_periods == (0, 1, 2)
    assert fold.test_periods == (3,)


def test_split_requires_enough_periods() -> None:
    examples = generate_synthetic_examples(periods=2, examples_per_period=2)
    with pytest.raises(ValueError, match="at least"):
        rolling_origin_folds(examples, minimum_train_periods=2)
