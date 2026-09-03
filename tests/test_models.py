import torch

from seqformer_lab.config import ModelConfig
from seqformer_lab.models import ConditionalTransformerVAE, MaskedSequenceTransformer, cvae_loss
from seqformer_lab.vocabulary import SequenceVocabulary


def tiny_config() -> ModelConfig:
    vocabulary = SequenceVocabulary(alphabet="ABCD")
    return ModelConfig(
        vocab_size=vocabulary.size,
        max_length=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=24,
        dropout=0.0,
        num_conditions=2,
        latent_dim=6,
    )


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocabulary = SequenceVocabulary(alphabet="ABCD")
    encoded = [vocabulary.encode(sequence, max_length=8) for sequence in ("ABCD", "DCBA")]
    ids = torch.tensor([item[0] for item in encoded])
    attention = torch.tensor([item[1] for item in encoded], dtype=torch.bool)
    conditions = torch.tensor([0, 1])
    return ids, attention, conditions


def test_masked_transformer_shape_and_tied_weights() -> None:
    model = MaskedSequenceTransformer(tiny_config())
    ids, attention, conditions = batch()
    logits = model(ids, attention, conditions)
    assert logits.shape == (2, 8, tiny_config().vocab_size)
    assert model.output.weight is model.embedding.token.weight


def test_conditional_vae_has_finite_gradients() -> None:
    model = ConditionalTransformerVAE(tiny_config())
    ids, attention, conditions = batch()
    logits, mean, log_variance = model(ids, attention, conditions)
    total, reconstruction, divergence = cvae_loss(
        logits,
        ids,
        mean,
        log_variance,
        pad_id=0,
    )
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(reconstruction)
    assert torch.isfinite(divergence)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_mean_latent_path_is_deterministic() -> None:
    model = ConditionalTransformerVAE(tiny_config()).eval()
    ids, attention, conditions = batch()
    with torch.no_grad():
        first = model(ids, attention, conditions, sample_latent=False)[0]
        second = model(ids, attention, conditions, sample_latent=False)[0]
    assert torch.equal(first, second)
