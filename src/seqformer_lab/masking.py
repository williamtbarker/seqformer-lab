"""BERT-style masking with explicit special-token protection."""

from __future__ import annotations

import torch
from torch import Tensor

from seqformer_lab.vocabulary import SequenceVocabulary


def mask_tokens(
    input_ids: Tensor,
    vocabulary: SequenceVocabulary,
    *,
    probability: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Apply 80/10/10 masking and return corrupted inputs plus sparse labels."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")

    eligible = input_ids.ge(vocabulary.first_regular_id)
    selected = (
        torch.rand(input_ids.shape, generator=generator, device="cpu").to(input_ids.device)
        < probability
    )
    selected &= eligible

    # A row with no supervised token makes tiny datasets unnecessarily unstable.
    for row in range(input_ids.shape[0]):
        if not bool(selected[row].any()) and bool(eligible[row].any()):
            first = int(torch.nonzero(eligible[row], as_tuple=False)[0].item())
            selected[row, first] = True

    labels = input_ids.clone()
    labels[~selected] = -100
    masked = input_ids.clone()
    decisions = torch.rand(input_ids.shape, generator=generator, device="cpu").to(input_ids.device)

    replace_with_mask = selected & decisions.lt(0.8)
    replace_randomly = selected & decisions.ge(0.8) & decisions.lt(0.9)
    masked[replace_with_mask] = vocabulary.mask_id
    random_ids = torch.randint(
        vocabulary.first_regular_id,
        vocabulary.size,
        input_ids.shape,
        generator=generator,
        device="cpu",
    ).to(input_ids.device)
    masked[replace_randomly] = random_ids[replace_randomly]
    return masked, labels
