import json
from pathlib import Path

import pytest

from seqformer_lab.data import generate_synthetic_examples, read_examples, write_examples


def test_synthetic_data_is_reproducible(tmp_path: Path) -> None:
    first = generate_synthetic_examples(seed=9, periods=3, examples_per_period=4)
    second = generate_synthetic_examples(seed=9, periods=3, examples_per_period=4)
    assert first == second
    assert len(first) == 12
    path = tmp_path / "examples.jsonl"
    first_hash = write_examples(path, first)
    second_hash = write_examples(tmp_path / "copy.jsonl", second)
    assert first_hash == second_hash
    assert read_examples(path) == first


def test_reader_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sequence": "ABC"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_examples(path)


@pytest.mark.parametrize("invalid_period", [1.5, "1", True])
def test_reader_rejects_non_integer_periods(tmp_path: Path, invalid_period: object) -> None:
    path = tmp_path / "bad-period.jsonl"
    record = {"sequence": "ACDEFG", "period": invalid_period, "condition": 0}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="period and condition must be integers"):
        read_examples(path)
