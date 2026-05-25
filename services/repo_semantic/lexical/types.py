"""Small lexical retrieval value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparseToken:
    """One observed lexical token plus its canonical vocabulary key."""

    surface: str
    canonical: str
    token_type: str


@dataclass(frozen=True, slots=True)
class EncodedSparseVector:
    """Qdrant-compatible sparse vector values without importing Qdrant here."""

    indices: list[int]
    values: list[float]
