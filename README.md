# SeqFormer Lab

[![CI](https://github.com/williamtbarker/seqformer-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/williamtbarker/seqformer-lab/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SeqFormer Lab is a compact, reproducible laboratory for transformer modeling of timestamped
symbolic sequences. It provides a condition-aware masked sequence model, a conditional
transformer variational autoencoder, leakage-resistant temporal splits, deterministic synthetic
fixtures, and inspectable experiment artifacts.

The included demonstration trains on generated motif-bearing sequences. It requires no private
dataset, pretrained checkpoint, external service, GPU, or network access after installation.

## Why this repository exists

Sequence-model experiments commonly become collections of training scripts tied to local CSVs,
large embedding caches, and undocumented split logic. SeqFormer Lab extracts the reusable parts
into explicit interfaces:

- a stable character vocabulary with protected special tokens;
- BERT-style 80/10/10 masking that always supervises at least one eligible token per row;
- expanding-window and final-period temporal evaluation with no future rows in training;
- a PyTorch encoder with condition embeddings and tied input/output token weights;
- a conditional transformer VAE with a tested reconstruction-plus-KL objective;
- atomic state-dictionary checkpoints loaded with `weights_only=True`;
- configuration, history, metrics, source hash, seed, device, and split periods recorded as JSON.

It is deliberately a research and engineering reference—not evidence that a model trained on the
synthetic example has scientific or predictive validity.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch selects the appropriate wheel for supported Intel or Apple Silicon Macs. The default
device is CPU so the demonstration is comparable across platforms.

## Run the complete demonstration

```bash
seqformer-lab demo --output artifacts/demo --epochs 2
```

This performs the entire workflow:

1. Generate five ordered periods of synthetic sequences with condition-specific motifs.
2. Hold the final period out and train only on earlier observations.
3. Train a small masked transformer on CPU.
4. Evaluate one deterministic masked view of the held-out period.
5. Write a model state dictionary and human-readable experiment metadata.

The output directory contains:

| Artifact | Contents |
|---|---|
| `dataset.jsonl` | Strict, portable sequence records |
| `model.pt` | State dictionary only—not a pickled model object |
| `model_config.json` | Complete architecture configuration |
| `vocabulary.json` | Alphabet required to interpret token IDs |
| `history.json` | Mean training loss by epoch |
| `metrics.json` | Held-out loss, perplexity, accuracy, and token count |
| `manifest.json` | Dataset SHA-256, seed, device, split periods, row counts, and training config |

Re-evaluate the checkpoint:

```bash
seqformer-lab evaluate \
  --data artifacts/demo/dataset.jsonl \
  --checkpoint artifacts/demo
```

## Use a custom dataset

Input is newline-delimited JSON with exactly three fields:

```json
{"condition":0,"period":0,"sequence":"ACDEFGHIKLMN"}
{"condition":1,"period":1,"sequence":"KLMNPQRSTVWY"}
```

`period` must be a non-negative ordered integer and `condition` a zero-based non-negative label.
The default vocabulary recognizes the 20 canonical amino-acid letters; other letters map to an
explicit unknown token. The final observed period is always held out by the `train` command.

```bash
seqformer-lab train \
  --data examples/sequences.jsonl \
  --output artifacts/custom \
  --epochs 5 \
  --batch-size 16 \
  --device cpu
```

## Models

### `MaskedSequenceTransformer`

The masked model combines learned token and position embeddings with an observation-level
condition embedding. Padding is excluded by an encoder key-padding mask. The normalized encoder
states are projected back to the vocabulary through weights tied to the input token embedding.

### `ConditionalTransformerVAE`

The VAE pools non-padding encoder states into latent mean and log-variance vectors. Its causal
transformer decoder receives the sampled latent vector and condition embedding at every position.
The public `cvae_loss` function combines next-token reconstruction loss with a configurable KL
term. Unit tests cover tensor shapes, deterministic mean-latent inference, backpropagation, and
finite gradients. The initial CLI concentrates on the masked-model workflow so the example stays
fast and directly reproducible.

## Temporal evaluation contract

For every fold:

```text
max(training period) < min(test period)
```

Optional gaps remove periods immediately preceding the test window. No random train/test split is
offered by the CLI because it would answer a different—and often misleading—question for evolving
sequence populations.

## Verification

```bash
./scripts/verify.sh
```

Verification runs formatting, linting, strict type checking, unit and integration tests, package
builds, a fresh CPU demonstration, and checkpoint re-evaluation. GitHub Actions repeats the suite
on Ubuntu and macOS.

## Project provenance

The design was reconstructed from several years of private experiments involving temporal masked
modeling, multi-input transformer fusion, conditional latent sequence models, embedding caches,
and explainability tooling. The public implementation was rewritten around synthetic data and
small deterministic tests. Private datasets, organization-specific paths, internal infrastructure,
trained weights, and unsupported performance claims are not included.

## Limitations

- The built-in tokenizer is character-level, not a pretrained protein-language-model tokenizer.
- Synthetic metrics prove execution and artifact reproducibility, not biological validity.
- CPU runs are the reproducibility baseline; MPS and CUDA may differ slightly numerically.
- The package does not download models or make network calls.

## License

MIT. See [LICENSE](LICENSE).
