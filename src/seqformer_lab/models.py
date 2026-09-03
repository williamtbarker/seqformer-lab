"""Compact transformer models for masked prediction and conditional generation."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from seqformer_lab.config import ModelConfig


class TokenPositionEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.position.num_embeddings:
            raise ValueError("sequence exceeds configured max_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        return cast(
            Tensor,
            self.dropout(
                self.token(input_ids) * math.sqrt(self.token.embedding_dim)
                + self.position(positions)
            ),
        )


def _encoder(config: ModelConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)


class MaskedSequenceTransformer(nn.Module):
    """Condition-aware encoder trained to recover masked sequence tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenPositionEmbedding(config)
        self.condition = nn.Embedding(config.num_conditions, config.d_model)
        self.encoder = _encoder(config)
        self.normalization = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.embedding.token.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        condition_ids: Tensor,
    ) -> Tensor:
        if condition_ids.ndim != 1 or condition_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("condition_ids must have shape [batch]")
        if bool(condition_ids.lt(0).any()) or bool(
            condition_ids.ge(self.config.num_conditions).any()
        ):
            raise ValueError("condition id outside configured range")
        hidden = self.embedding(input_ids)
        hidden = hidden + self.condition(condition_ids).unsqueeze(1)
        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        return cast(Tensor, self.output(self.normalization(encoded)))


class ConditionalTransformerVAE(nn.Module):
    """A conditional variational autoencoder with transformer encoder and causal decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenPositionEmbedding(config)
        self.condition = nn.Embedding(config.num_conditions, config.d_model)
        self.encoder = _encoder(config)
        self.to_mean = nn.Linear(config.d_model, config.latent_dim)
        self.to_log_variance = nn.Linear(config.d_model, config.latent_dim)
        self.latent_projection = nn.Linear(config.latent_dim, config.d_model)
        self.decoder = _encoder(config)
        self.normalization = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def encode(
        self, input_ids: Tensor, attention_mask: Tensor, condition_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        hidden = self.embedding(input_ids) + self.condition(condition_ids).unsqueeze(1)
        encoded = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        weights = attention_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.to_mean(pooled), self.to_log_variance(pooled)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor, *, sample: bool) -> Tensor:
        if not sample:
            return mean
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode(
        self,
        decoder_input_ids: Tensor,
        attention_mask: Tensor,
        condition_ids: Tensor,
        latent: Tensor,
    ) -> Tensor:
        hidden = self.embedding(decoder_input_ids)
        hidden = hidden + self.condition(condition_ids).unsqueeze(1)
        hidden = hidden + self.latent_projection(latent).unsqueeze(1)
        length = decoder_input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask.bool(),
            is_causal=True,
        )
        return cast(Tensor, self.output(self.normalization(decoded)))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        condition_ids: Tensor,
        *,
        sample_latent: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(input_ids, attention_mask, condition_ids)
        latent = self.reparameterize(mean, log_variance, sample=sample_latent)
        logits = self.decode(input_ids, attention_mask, condition_ids, latent)
        return logits, mean, log_variance


def cvae_loss(
    logits: Tensor,
    targets: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    *,
    pad_id: int,
    beta: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, reconstruction, and KL losses for a conditional VAE batch."""

    reconstruction = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        targets[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )
    divergence = -0.5 * torch.mean(1.0 + log_variance - mean.square() - log_variance.exp())
    return reconstruction + beta * divergence, reconstruction, divergence
