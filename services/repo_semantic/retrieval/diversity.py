"""Result diversity helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def apply_max_results_per_file(
    ranked: list[T],
    *,
    top_k: int,
    max_results_per_file: int | None,
    relative_path: Callable[[T], str],
) -> list[T]:
    """Keep global score ordering while limiting chunks per file."""

    if max_results_per_file is not None and max_results_per_file <= 0:
        return []
    kept: list[T] = []
    per_file: dict[str, int] = {}
    for item in ranked:
        path = relative_path(item)
        if max_results_per_file is not None and per_file.get(path, 0) >= max_results_per_file:
            continue
        kept.append(item)
        per_file[path] = per_file.get(path, 0) + 1
        if len(kept) >= top_k:
            break
    return kept
