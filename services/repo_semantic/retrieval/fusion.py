"""Rank-fusion helpers for retrieval branches."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord

from services.repo_semantic.retrieval.constants import RRF_K


def weighted_rrf(
    branches: dict[str, list[tuple[ChunkRecord, float]]],
    weights: dict[str, float],
    *,
    rrf_k: int = RRF_K,
) -> list[tuple[ChunkRecord, float, float, float, dict[str, int]]]:
    """Fuse branch ranks with weighted reciprocal rank fusion."""

    fused: dict[str, dict[str, object]] = {}
    for branch_name, ranked in branches.items():
        weight = weights.get(branch_name, 1.0)
        for rank, (chunk, raw_score) in enumerate(ranked, start=1):
            entry = fused.setdefault(
                chunk.point_id,
                {
                    "chunk": chunk,
                    "score": 0.0,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "branch_ranks": {},
                },
            )
            entry["score"] = float(entry["score"]) + weight / (rrf_k + rank)
            if branch_name == "dense":
                entry["dense_score"] = max(float(entry["dense_score"]), raw_score)
            if branch_name == "sparse":
                entry["sparse_score"] = max(float(entry["sparse_score"]), raw_score)
            branch_ranks = entry["branch_ranks"]
            assert isinstance(branch_ranks, dict)
            branch_ranks[branch_name] = rank

    rows: list[tuple[ChunkRecord, float, float, float, dict[str, int]]] = []
    for entry in fused.values():
        chunk = entry["chunk"]
        assert isinstance(chunk, ChunkRecord)
        branch_ranks = entry["branch_ranks"]
        assert isinstance(branch_ranks, dict)
        rows.append(
            (
                chunk,
                float(entry["score"]),
                float(entry["dense_score"]),
                float(entry["sparse_score"]),
                {str(key): int(value) for key, value in branch_ranks.items()},
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            item[1],
            -min(item[4].values()) if item[4] else 0,
        ),
        reverse=True,
    )
