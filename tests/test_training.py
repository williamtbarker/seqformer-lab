from pathlib import Path

import pytest
import torch

from seqformer_lab.artifacts import load_masked_model, save_state_dict, write_json
from seqformer_lab.config import ModelConfig, TrainingConfig
from seqformer_lab.data import generate_synthetic_examples
from seqformer_lab.training import evaluate_masked_model, train_masked_model
from seqformer_lab.vocabulary import SequenceVocabulary


def test_tiny_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    vocabulary = SequenceVocabulary(alphabet="ABCDEFGH")
    examples = generate_synthetic_examples(
        periods=3,
        examples_per_period=4,
        sequence_length=8,
        conditions=2,
        seed=3,
        alphabet=vocabulary.alphabet,
    )
    model_config = ModelConfig(
        vocab_size=vocabulary.size,
        max_length=10,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=24,
        dropout=0.0,
        num_conditions=2,
        latent_dim=4,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        mask_probability=0.25,
        seed=3,
    )
    device = torch.device("cpu")
    model, history = train_masked_model(
        examples,
        vocabulary,
        model_config,
        training_config,
        device=device,
    )
    assert len(history) == 1
    assert history[0] > 0
    metrics = evaluate_masked_model(model, examples, vocabulary, seed=8, device=device)
    assert metrics.loss > 0
    assert 0.0 <= metrics.masked_accuracy <= 1.0

    write_json(tmp_path / "model_config.json", model_config.to_dict())
    write_json(tmp_path / "vocabulary.json", vocabulary.to_dict())
    save_state_dict(tmp_path / "model.pt", model)
    loaded, loaded_vocabulary = load_masked_model(tmp_path, device=device)
    assert loaded_vocabulary == vocabulary
    loaded_metrics = evaluate_masked_model(loaded, examples, vocabulary, seed=8, device=device)
    assert loaded_metrics.loss == pytest.approx(metrics.loss)
