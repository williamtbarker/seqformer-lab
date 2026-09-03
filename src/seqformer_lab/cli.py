"""Command-line interface for reproducible sequence-model experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from seqformer_lab import __version__
from seqformer_lab.artifacts import (
    load_masked_model,
    prepare_output_directory,
    save_state_dict,
    write_json,
)
from seqformer_lab.config import ModelConfig, TrainingConfig
from seqformer_lab.data import (
    SequenceExample,
    dataset_sha256,
    generate_synthetic_examples,
    read_examples,
    write_examples,
)
from seqformer_lab.splits import final_temporal_holdout
from seqformer_lab.training import (
    evaluate_masked_model,
    resolve_device,
    select_examples,
    train_masked_model,
)
from seqformer_lab.vocabulary import SequenceVocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seqformer-lab",
        description="Reproducible transformer experiments for timestamped symbolic sequences.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-data", help="write a deterministic synthetic dataset"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--periods", type=int, default=5)
    generate.add_argument("--examples-per-period", type=int, default=20)
    generate.add_argument("--sequence-length", type=int, default=14)
    generate.add_argument("--conditions", type=int, default=3)
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train", help="train and evaluate a compact masked transformer")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-3)
    train.add_argument("--mask-probability", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    train.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved model on a JSONL dataset")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")

    demo = subparsers.add_parser("demo", help="generate data, train, evaluate, and save artifacts")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--epochs", type=int, default=2)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    demo.add_argument("--force", action="store_true")
    return parser


def _infer_model_config(
    examples: Sequence[SequenceExample], vocabulary: SequenceVocabulary
) -> ModelConfig:
    sequences = [example.sequence for example in examples]
    conditions = [example.condition for example in examples]
    return ModelConfig(
        vocab_size=vocabulary.size,
        max_length=max(len(sequence) for sequence in sequences) + 2,
        num_conditions=max(conditions) + 1,
    )


def _train_command(
    *,
    data_path: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_probability: float,
    seed: int,
    device_name: str,
    force: bool,
) -> dict[str, object]:
    prepare_output_directory(output, force=force)
    examples = read_examples(data_path)
    vocabulary = SequenceVocabulary()
    model_config = _infer_model_config(examples, vocabulary)
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_probability=mask_probability,
        seed=seed,
    )
    fold = final_temporal_holdout(examples)
    train_examples = select_examples(examples, fold.train_indices)
    test_examples = select_examples(examples, fold.test_indices)
    device = resolve_device(device_name)
    model, history = train_masked_model(
        train_examples,
        vocabulary,
        model_config,
        training_config,
        device=device,
    )
    metrics = evaluate_masked_model(
        model,
        test_examples,
        vocabulary,
        batch_size=batch_size,
        mask_probability=mask_probability,
        seed=seed + 10_000,
        device=device,
    )

    write_json(output / "model_config.json", model_config.to_dict())
    write_json(output / "vocabulary.json", vocabulary.to_dict())
    write_json(output / "history.json", {"training_loss": history})
    write_json(output / "metrics.json", metrics.to_dict())
    save_state_dict(output / "model.pt", model)
    manifest: dict[str, object] = {
        "dataset_sha256": dataset_sha256(data_path),
        "device": str(device),
        "seed": seed,
        "train_periods": list(fold.train_periods),
        "test_periods": list(fold.test_periods),
        "train_rows": len(train_examples),
        "test_rows": len(test_examples),
        "training": training_config.to_dict(),
    }
    write_json(output / "manifest.json", manifest)
    return {"artifacts": str(output), "metrics": metrics.to_dict(), "manifest": manifest}


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "generate-data":
        if arguments.output.exists() and not arguments.force:
            raise FileExistsError(f"output already exists: {arguments.output}")
        examples = generate_synthetic_examples(
            periods=arguments.periods,
            examples_per_period=arguments.examples_per_period,
            sequence_length=arguments.sequence_length,
            conditions=arguments.conditions,
            seed=arguments.seed,
        )
        digest = write_examples(arguments.output, examples)
        return {"output": str(arguments.output), "rows": len(examples), "sha256": digest}

    if arguments.command == "train":
        return _train_command(
            data_path=arguments.data,
            output=arguments.output,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            mask_probability=arguments.mask_probability,
            seed=arguments.seed,
            device_name=arguments.device,
            force=arguments.force,
        )

    if arguments.command == "evaluate":
        examples = read_examples(arguments.data)
        fold = final_temporal_holdout(examples)
        test_examples = select_examples(examples, fold.test_indices)
        device = resolve_device(arguments.device)
        model, vocabulary = load_masked_model(arguments.checkpoint, device=device)
        metrics = evaluate_masked_model(
            model,
            test_examples,
            vocabulary,
            seed=10_017,
            device=device,
        )
        return {"metrics": metrics.to_dict(), "test_periods": list(fold.test_periods)}

    if arguments.command == "demo":
        prepare_output_directory(arguments.output, force=arguments.force)
        data_path = arguments.output / "dataset.jsonl"
        examples = generate_synthetic_examples(
            periods=5,
            examples_per_period=12,
            sequence_length=14,
            conditions=3,
            seed=arguments.seed,
        )
        write_examples(data_path, examples)
        return _train_command(
            data_path=data_path,
            output=arguments.output,
            epochs=arguments.epochs,
            batch_size=12,
            learning_rate=3e-3,
            mask_probability=0.20,
            seed=arguments.seed,
            device_name=arguments.device,
            force=True,
        )
    raise AssertionError(f"unhandled command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        result = run(parser.parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
