"""Atomic, inspectable experiment artifact handling."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from seqformer_lab.config import ModelConfig
from seqformer_lab.models import MaskedSequenceTransformer
from seqformer_lab.vocabulary import SequenceVocabulary

KNOWN_ARTIFACTS = {
    "dataset.jsonl",
    "history.json",
    "manifest.json",
    "metrics.json",
    "model.pt",
    "model_config.json",
    "vocabulary.json",
}


def prepare_output_directory(path: Path, *, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = {item.name for item in path.iterdir()}
    if existing and not force:
        raise FileExistsError(f"output directory is not empty: {path}")
    unexpected = existing - KNOWN_ARTIFACTS
    if unexpected and force:
        names = ", ".join(sorted(unexpected))
        raise FileExistsError(f"refusing to overwrite directory with unrelated files: {names}")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def save_state_dict(path: Path, model: MaskedSequenceTransformer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(model.state_dict(), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_masked_model(
    directory: Path, *, device: torch.device
) -> tuple[MaskedSequenceTransformer, SequenceVocabulary]:
    config_data = read_json(directory / "model_config.json")
    vocabulary = SequenceVocabulary.from_dict(read_json(directory / "vocabulary.json"))
    config = ModelConfig(**config_data)
    if config.vocab_size != vocabulary.size:
        raise ValueError("checkpoint vocabulary and model_config disagree")
    model = MaskedSequenceTransformer(config)
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, vocabulary
