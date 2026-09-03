"""Synthetic fixtures and JSONL I/O for timestamped sequence examples."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SequenceExample:
    """One sequence observation with an ordered period and condition label."""

    sequence: str
    period: int
    condition: int

    def __post_init__(self) -> None:
        if not self.sequence or not self.sequence.isalpha():
            raise ValueError("sequence must be non-empty alphabetic text")
        if self.period < 0 or self.condition < 0:
            raise ValueError("period and condition must be non-negative")


def generate_synthetic_examples(
    *,
    periods: int = 5,
    examples_per_period: int = 20,
    sequence_length: int = 14,
    conditions: int = 3,
    seed: int = 17,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
) -> list[SequenceExample]:
    """Generate reproducible motif-bearing sequences with gradual temporal drift."""

    if periods < 2 or examples_per_period <= 0 or sequence_length < 6:
        raise ValueError("need at least two periods, one example per period, and length >= 6")
    if conditions <= 0 or conditions > len(alphabet):
        raise ValueError("conditions must fit within the alphabet")

    rng = random.Random(seed)
    examples: list[SequenceExample] = []
    motif_width = min(4, sequence_length // 2)
    for period in range(periods):
        for row in range(examples_per_period):
            condition = row % conditions
            sequence = [rng.choice(alphabet) for _ in range(sequence_length)]
            start = (condition * 3) % (sequence_length - motif_width + 1)
            motif = [
                alphabet[(condition * motif_width + offset) % len(alphabet)]
                for offset in range(motif_width)
            ]
            motif[period % motif_width] = alphabet[(condition + period + 7) % len(alphabet)]
            sequence[start : start + motif_width] = motif
            noise_index = rng.randrange(sequence_length)
            if rng.random() < 0.15:
                sequence[noise_index] = rng.choice(alphabet)
            examples.append(
                SequenceExample(
                    sequence="".join(sequence),
                    period=period,
                    condition=condition,
                )
            )
    return examples


def write_examples(path: Path, examples: Iterable[SequenceExample]) -> str:
    """Write canonical JSONL atomically and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            line = json.dumps(asdict(example), sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    temporary.replace(path)
    return digest.hexdigest()


def read_examples(path: Path) -> list[SequenceExample]:
    """Read and validate the strict JSONL interchange format."""

    examples: list[SequenceExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record is not an object")
                if set(record) != {"sequence", "period", "condition"}:
                    raise ValueError("record keys must be sequence, period, and condition")
                sequence = record["sequence"]
                period = record["period"]
                condition = record["condition"]
                if not isinstance(sequence, str):
                    raise TypeError("sequence must be a string")
                if (
                    not isinstance(period, int)
                    or isinstance(period, bool)
                    or not isinstance(condition, int)
                    or isinstance(condition, bool)
                ):
                    raise TypeError("period and condition must be integers")
                examples.append(
                    SequenceExample(
                        sequence=sequence.upper(),
                        period=period,
                        condition=condition,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid record at line {line_number}: {error}") from error
    if not examples:
        raise ValueError("dataset is empty")
    return examples


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
