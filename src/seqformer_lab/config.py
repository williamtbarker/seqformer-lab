"""Validated configuration objects shared by models and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture parameters for the compact sequence transformers."""

    vocab_size: int
    max_length: int = 24
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 96
    dropout: float = 0.1
    num_conditions: int = 3
    latent_dim: int = 16

    def __post_init__(self) -> None:
        if self.vocab_size < 6:
            raise ValueError("vocab_size must include five special tokens and an alphabet")
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive")
        if self.d_model <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers <= 0 or self.dim_feedforward <= 0:
            raise ValueError("num_layers and dim_feedforward must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_conditions <= 0 or self.latent_dim <= 0:
            raise ValueError("num_conditions and latent_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Parameters for deterministic masked-language-model training."""

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-3
    mask_probability: float = 0.20
    seed: int = 17
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
