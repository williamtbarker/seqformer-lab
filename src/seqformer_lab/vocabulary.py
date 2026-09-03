"""A small explicit vocabulary for character-level symbolic sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class SequenceVocabulary:
    """Encode one-character tokens with stable special-token identifiers."""

    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"

    PAD: ClassVar[str] = "[PAD]"
    MASK: ClassVar[str] = "[MASK]"
    BOS: ClassVar[str] = "[BOS]"
    EOS: ClassVar[str] = "[EOS]"
    UNK: ClassVar[str] = "[UNK]"
    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = (PAD, MASK, BOS, EOS, UNK)

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise ValueError("alphabet cannot be empty")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet tokens must be unique")
        if any(len(token) != 1 or token.isspace() for token in self.alphabet):
            raise ValueError("alphabet must contain unique, non-whitespace characters")

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.SPECIAL_TOKENS + tuple(self.alphabet)

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def mask_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def unk_id(self) -> int:
        return 4

    @property
    def first_regular_id(self) -> int:
        return len(self.SPECIAL_TOKENS)

    def encode(self, sequence: str, *, max_length: int) -> tuple[list[int], list[int]]:
        """Return padded token IDs and an attention mask, including BOS/EOS."""

        normalized = sequence.strip().upper()
        if not normalized:
            raise ValueError("sequence cannot be empty")
        if len(normalized) + 2 > max_length:
            raise ValueError(
                f"sequence needs {len(normalized) + 2} tokens but max_length is {max_length}"
            )
        mapping = {token: index for index, token in enumerate(self.tokens)}
        ids = [self.bos_id]
        ids.extend(mapping.get(token, self.unk_id) for token in normalized)
        ids.append(self.eos_id)
        attention = [1] * len(ids)
        padding = max_length - len(ids)
        ids.extend([self.pad_id] * padding)
        attention.extend([0] * padding)
        return ids, attention

    def decode(self, token_ids: list[int], *, include_unknown: bool = True) -> str:
        """Decode regular tokens and optionally preserve unknowns as ``?``."""

        output: list[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if token_id < self.first_regular_id:
                if token_id == self.unk_id and include_unknown:
                    output.append("?")
                continue
            if token_id >= self.size:
                raise ValueError(f"token id out of range: {token_id}")
            output.append(self.tokens[token_id])
        return "".join(output)

    def to_dict(self) -> dict[str, str]:
        return {"alphabet": self.alphabet}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SequenceVocabulary:
        alphabet = data.get("alphabet")
        if not isinstance(alphabet, str):
            raise ValueError("vocabulary alphabet must be a string")
        return cls(alphabet=alphabet)
