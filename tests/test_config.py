import pytest

from seqformer_lab.config import ModelConfig, TrainingConfig


def test_model_configuration_rejects_zero_attention_heads() -> None:
    with pytest.raises(ValueError, match="nhead must be positive"):
        ModelConfig(vocab_size=25, nhead=0)


def test_model_configuration_requires_even_head_width() -> None:
    with pytest.raises(ValueError, match="divisible by nhead"):
        ModelConfig(vocab_size=25, d_model=30, nhead=4)


def test_training_configuration_validates_mask_probability() -> None:
    with pytest.raises(ValueError, match="mask_probability"):
        TrainingConfig(mask_probability=0.0)
