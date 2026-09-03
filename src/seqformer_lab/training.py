"""Deterministic CPU-friendly training and evaluation routines."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from seqformer_lab.config import ModelConfig, TrainingConfig
from seqformer_lab.data import SequenceExample
from seqformer_lab.masking import mask_tokens
from seqformer_lab.models import MaskedSequenceTransformer
from seqformer_lab.vocabulary import SequenceVocabulary


class EncodedSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        examples: Sequence[SequenceExample],
        vocabulary: SequenceVocabulary,
        max_length: int,
    ) -> None:
        encoded = [
            vocabulary.encode(example.sequence, max_length=max_length) for example in examples
        ]
        self.input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.attention_masks = torch.tensor([item[1] for item in encoded], dtype=torch.bool)
        self.conditions = torch.tensor(
            [example.condition for example in examples], dtype=torch.long
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.input_ids[index], self.attention_masks[index], self.conditions[index]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    loss: float
    perplexity: float
    masked_accuracy: float
    masked_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "masked_accuracy": self.masked_accuracy,
            "masked_tokens": self.masked_tokens,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return device


def _loader(
    dataset: EncodedSequenceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def train_masked_model(
    examples: Sequence[SequenceExample],
    vocabulary: SequenceVocabulary,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    device: torch.device,
) -> tuple[MaskedSequenceTransformer, list[float]]:
    """Train a masked model and return the model plus mean loss per epoch."""

    if not examples:
        raise ValueError("training examples cannot be empty")
    seed_everything(training_config.seed)
    dataset = EncodedSequenceDataset(examples, vocabulary, model_config.max_length)
    loader = _loader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    )
    model = MaskedSequenceTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    losses: list[float] = []

    for epoch in range(training_config.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        masking_generator = torch.Generator().manual_seed(training_config.seed + epoch)
        for input_ids, attention_mask, conditions in loader:
            masked, labels = mask_tokens(
                input_ids,
                vocabulary,
                probability=training_config.mask_probability,
                generator=masking_generator,
            )
            masked = masked.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            conditions = conditions.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(masked, attention_mask, conditions)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch stub limitation
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(total_loss / batches)
    return model, losses


@torch.no_grad()
def evaluate_masked_model(
    model: MaskedSequenceTransformer,
    examples: Sequence[SequenceExample],
    vocabulary: SequenceVocabulary,
    *,
    batch_size: int = 32,
    mask_probability: float = 0.20,
    seed: int = 101,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate against one deterministic masked view of the supplied examples."""

    dataset = EncodedSequenceDataset(examples, vocabulary, model.config.max_length)
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    for input_ids, attention_mask, conditions in loader:
        masked, labels = mask_tokens(
            input_ids,
            vocabulary,
            probability=mask_probability,
            generator=generator,
        )
        logits = model(masked.to(device), attention_mask.to(device), conditions.to(device))
        labels = labels.to(device)
        selected = labels.ne(-100)
        loss_sum = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss_sum.cpu())
        total_correct += int((logits.argmax(dim=-1)[selected] == labels[selected]).sum().cpu())
        total_masked += int(selected.sum().cpu())
    if total_masked == 0:
        raise RuntimeError("evaluation selected no tokens")
    mean_loss = total_loss / total_masked
    return EvaluationMetrics(
        loss=mean_loss,
        perplexity=math.exp(min(mean_loss, 50.0)),
        masked_accuracy=total_correct / total_masked,
        masked_tokens=total_masked,
    )


def select_examples(
    examples: Sequence[SequenceExample], indices: Iterable[int]
) -> list[SequenceExample]:
    return [examples[index] for index in indices]
