import pytest

from seqformer_lab.vocabulary import SequenceVocabulary


def test_round_trip_and_padding() -> None:
    vocabulary = SequenceVocabulary(alphabet="ABCD")
    token_ids, attention = vocabulary.encode("ABDC", max_length=8)
    assert vocabulary.decode(token_ids) == "ABDC"
    assert attention == [1, 1, 1, 1, 1, 1, 0, 0]


def test_unknown_token_is_explicit() -> None:
    vocabulary = SequenceVocabulary(alphabet="AB")
    token_ids, _ = vocabulary.encode("AX", max_length=4)
    assert vocabulary.decode(token_ids) == "A?"


def test_rejects_overlong_sequence() -> None:
    with pytest.raises(ValueError, match="max_length"):
        SequenceVocabulary().encode("ACDE", max_length=5)
