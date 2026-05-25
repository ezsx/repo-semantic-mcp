"""Dense/sparse retrieval status helpers."""

from __future__ import annotations

from collections.abc import Callable

from services.repo_semantic.contracts.status import (
    RetrievalContractStatus,
    RetrievalScopeStatus,
    StatusAction,
    StatusWarning,
)
from services.repo_semantic.retrieval.constants import FILTER_PAYLOAD_FIELDS
from services.repo_semantic.lexical import LEXICAL_ANALYZER_VERSION, SPARSE_ENCODER_KIND


def retrieval_scope_status(
    *,
    scope: str,
    points_count: int,
    store,
    read_collection_exists: Callable[[str], bool],
    current_collection_exists: Callable[[str], bool],
    dense_schema_kind_for_scope: Callable[[str], str],
    load_sparse_manifest,
    sparse_manifest_for_scope,
    sparse_manifest_unavailable_codes,
    store_sparse_available: Callable[[str], bool],
    payload_index_statuses,
) -> RetrievalScopeStatus:
    """Build dense/sparse retrieval status for one logical scope."""

    exists = read_collection_exists(scope)
    current_exists = current_collection_exists(scope)
    dense_schema_kind = dense_schema_kind_for_scope(scope)
    dense_available = exists and dense_schema_kind in {"named", "unnamed_legacy", "unknown"}
    manifest = load_sparse_manifest(scope)
    compatible_manifest = sparse_manifest_for_scope(scope)
    qdrant_sparse_available = store_sparse_available(scope)
    unavailable_codes: list[str] = []
    state = "unknown"

    if manifest is not None and manifest.intentionally_empty and points_count == 0:
        state = "intentionally_empty"
    elif not exists:
        state = "missing"
        unavailable_codes.append("collection_missing")
    elif not current_exists:
        state = "legacy_dense_only"
        unavailable_codes.append("legacy_dense_only_collection")
        if manifest is not None:
            unavailable_codes.extend(sparse_manifest_unavailable_codes(manifest))
    elif compatible_manifest is None:
        state = "legacy_dense_only"
        unavailable_codes.extend(sparse_manifest_unavailable_codes(manifest))
    elif not qdrant_sparse_available:
        state = "legacy_dense_only"
        unavailable_codes.append("sparse_vector_missing")
    else:
        state = "ready"

    sparse_contract_compatible = compatible_manifest is not None
    sparse_available = (
        state == "ready"
        and qdrant_sparse_available
        and compatible_manifest is not None
        and not compatible_manifest.intentionally_empty
    )
    if dense_schema_kind == "unnamed_legacy":
        unavailable_codes.append("legacy_dense_only_collection")

    return RetrievalScopeStatus(
        scope=scope,  # type: ignore[arg-type]
        collection_name=store.collection_name(scope),
        state=state,  # type: ignore[arg-type]
        points_count=points_count,
        dense_available=dense_available,
        dense_schema_kind=dense_schema_kind  # type: ignore[arg-type]
        if dense_schema_kind in {"named", "unnamed_legacy", "missing", "unknown"}
        else "unknown",
        sparse_available=sparse_available,
        sparse_contract_compatible=sparse_contract_compatible,
        sparse_stats_stale=bool(compatible_manifest and compatible_manifest.sparse_stats_stale),
        sparse_manifest_hash=compatible_manifest.manifest_content_hash
        if compatible_manifest is not None
        else None,
        sparse_vocabulary_hash=compatible_manifest.vocabulary_hash
        if compatible_manifest is not None
        else None,
        sparse_corpus_stats_hash=compatible_manifest.corpus_stats_hash
        if compatible_manifest is not None
        else None,
        expected_lexical_analyzer_version=LEXICAL_ANALYZER_VERSION,
        stored_lexical_analyzer_version=manifest.lexical_analyzer_version
        if manifest is not None
        else None,
        expected_sparse_encoder_kind=SPARSE_ENCODER_KIND,
        stored_sparse_encoder_kind=manifest.sparse_encoder_kind
        if manifest is not None
        else None,
        payload_indexes=payload_index_statuses(scope),
        unavailable_codes=sorted(set(unavailable_codes)),
    )


def retrieval_status(
    *,
    points_by_scope: dict[str, int],
    scope_status_for_scope: Callable[[str, int], RetrievalScopeStatus],
    payload_fields_for_scope: Callable[[str], set[str]],
) -> RetrievalContractStatus:
    """Build global dense/sparse retrieval status."""

    scope_statuses = [
        scope_status_for_scope(scope, points_by_scope[scope])
        for scope in ("code", "docs")
    ]
    non_empty_scopes = [
        status
        for status in scope_statuses
        if status.points_count > 0 and status.state != "intentionally_empty"
    ]
    dense_available = any(status.dense_available for status in non_empty_scopes)
    sparse_available = bool(non_empty_scopes) and all(
        status.sparse_available for status in non_empty_scopes
    )
    sparse_contract_compatible = bool(non_empty_scopes) and all(
        status.sparse_contract_compatible for status in non_empty_scopes
    )
    sparse_unavailable_codes = sorted(
        {
            code
            for status in non_empty_scopes
            for code in status.unavailable_codes
            if code
            in {
                "sparse_manifest_missing",
                "sparse_analyzer_version_mismatch",
                "sparse_encoder_kind_mismatch",
                "sparse_contract_mismatch",
                "sparse_vector_missing",
                "legacy_dense_only_collection",
            }
        }
    )
    representative = next(
        (
            status
            for status in scope_statuses
            if status.sparse_manifest_hash is not None
        ),
        None,
    )
    stored_analyzer_versions = sorted(
        {
            status.stored_lexical_analyzer_version
            for status in non_empty_scopes
            if status.stored_lexical_analyzer_version is not None
        }
    )
    stored_encoder_kinds = sorted(
        {
            status.stored_sparse_encoder_kind
            for status in non_empty_scopes
            if status.stored_sparse_encoder_kind is not None
        }
    )
    payload_filter_pushdown_supported = bool(non_empty_scopes) and all(
        FILTER_PAYLOAD_FIELDS.issubset(payload_fields_for_scope(status.scope))
        for status in non_empty_scopes
    )
    return RetrievalContractStatus(
        dense_available=dense_available,
        sparse_available=sparse_available,
        sparse_contract_compatible=sparse_contract_compatible,
        payload_filter_pushdown_supported=payload_filter_pushdown_supported,
        legacy_lexical_fallback_enabled=True,
        unavailable_codes=sparse_unavailable_codes,
        sparse_unavailable_codes=sparse_unavailable_codes,
        sparse_manifest_hash=representative.sparse_manifest_hash
        if representative is not None
        else None,
        sparse_vocabulary_hash=representative.sparse_vocabulary_hash
        if representative is not None
        else None,
        sparse_corpus_stats_hash=representative.sparse_corpus_stats_hash
        if representative is not None
        else None,
        sparse_stats_stale=any(status.sparse_stats_stale for status in scope_statuses),
        expected_lexical_analyzer_version=LEXICAL_ANALYZER_VERSION,
        stored_lexical_analyzer_version=stored_analyzer_versions[0]
        if len(stored_analyzer_versions) == 1
        else ("mixed" if stored_analyzer_versions else None),
        expected_sparse_encoder_kind=SPARSE_ENCODER_KIND,
        stored_sparse_encoder_kind=stored_encoder_kinds[0]
        if len(stored_encoder_kinds) == 1
        else ("mixed" if stored_encoder_kinds else None),
        scope_statuses=scope_statuses,
    )


def retrieval_warnings_and_actions(
    *,
    retrieval_status: RetrievalContractStatus,
    total_points: int,
) -> tuple[list[StatusWarning], list[StatusAction]]:
    """Build status warnings/actions for sparse retrieval readiness."""

    warnings: list[StatusWarning] = []
    actions: list[StatusAction] = []
    if retrieval_status.sparse_unavailable_codes and total_points > 0:
        warnings.append(
            StatusWarning(
                code="sparse_branch_unavailable",
                severity="warning",
                detail="Dense search remains available, but full Qdrant dense+sparse hybrid requires an explicit rebuild.",
            )
        )
        actions.append(
            StatusAction(
                code="explicit_sparse_rebuild_recommended",
                severity="warning",
                title="Rebuild index to enable Qdrant sparse vectors",
                detail="Legacy dense-only or sparse-incompatible index state detected.",
                tool_hint="build_index(force_rebuild=true)",
            )
        )
    elif retrieval_status.sparse_stats_stale:
        warnings.append(
            StatusWarning(
                code="sparse_stats_stale",
                severity="warning",
                detail="Sparse BM25 corpus statistics are stale; hybrid search remains available with degraded lexical weighting.",
            )
        )
    return warnings, actions
