import torch

from seqformer_lab.masking import mask_tokens
from seqformer_lab.vocabulary import SequenceVocabulary


def test_masking_is_reproducible_and_protects_special_tokens() -> None:
    vocabulary = SequenceVocabulary(alphabet="ABCD")
    ids, _ = vocabulary.encode("ABCD", max_length=8)
    inputs = torch.tensor([ids, ids], dtype=torch.long)
    first = mask_tokens(
        inputs,
        vocabulary,
        probability=0.3,
        generator=torch.Generator().manual_seed(5),
    )
    second = mask_tokens(
        inputs,
        vocabulary,
        probability=0.3,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.all(first[1][:, 0] == -100)
    assert torch.all(first[1][:, 5:] == -100)
    assert torch.all(first[1].ne(-100).sum(dim=1) >= 1)
