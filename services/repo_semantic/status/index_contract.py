"""Index contract compatibility helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.contracts.status import IndexCollectionContract


@dataclass(frozen=True, slots=True)
class IndexContractSummary:
    """Summary over per-collection index contract statuses."""

    compatibility: str
    blocking: bool
    warning_codes: list[str]
    incompatibility_codes: list[str]


def sha256_text(value: str) -> str:
    """Return lowercase SHA-256 for contract string values."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def index_contract_issue(
    *,
    store,
    settings: SemanticMcpSettings,
    embedding_backend: str,
    embedding_model: str,
) -> str | None:
    """Return a blocking runtime/index compatibility issue if one exists."""

    expected_schema_version = str(settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION)
    expected_query_template_hash = sha256_text(settings.SEMANTIC_MCP_QUERY_TEMPLATE)
    expected_document_prefix_hash = sha256_text(settings.SEMANTIC_MCP_DOCUMENT_PREFIX)
    for scope in ("code", "docs"):
        contract = store.collection_embedding_contract(scope)
        if contract is None:
            continue
        if (
            contract.get("embedding_backend", "") != embedding_backend
            or contract.get("embedding_model", "") != embedding_model
        ):
            return (
                "indexed vectors were built with a different embedding backend or model; "
                "explicit rebuild is required"
            )
        stored_schema_version = contract.get("index_schema_version")
        if stored_schema_version is not None and str(stored_schema_version) != expected_schema_version:
            try:
                stored_schema_number = int(str(stored_schema_version))
                expected_schema_number = int(expected_schema_version)
            except ValueError:
                stored_schema_number = expected_schema_number = 0
            if stored_schema_number > expected_schema_number:
                return (
                    "indexed vectors were built with a newer index schema version; "
                    "explicit rebuild or runtime upgrade is required"
                )
        stored_query_template_hash = contract.get("query_template_hash")
        if (
            stored_query_template_hash
            and stored_query_template_hash != expected_query_template_hash
        ):
            return (
                "indexed vectors were built with a different query template; "
                "explicit rebuild is required"
            )
        stored_document_prefix_hash = contract.get("document_prefix_hash")
        if (
            stored_document_prefix_hash
            and stored_document_prefix_hash != expected_document_prefix_hash
        ):
            return (
                "indexed vectors were built with a different document prefix; "
                "explicit rebuild is required"
            )
    return None


def collection_contract(
    *,
    store,
    settings: SemanticMcpSettings,
    scope: str,
    points_count: int,
    lexical_documents: int | None,
    runtime_embedding_backend: str,
    runtime_embedding_model: str,
    runtime_query_template_hash: str,
    runtime_document_prefix_hash: str,
    read_collection_exists,
) -> IndexCollectionContract:
    """Build per-collection runtime-vs-stored compatibility status."""

    exists = read_collection_exists(scope)
    stored_contract = store.collection_embedding_contract(scope) if exists else None
    warning_codes: list[str] = []
    blocking = False
    compatibility = "compatible"
    stored_schema_version: int | str | None = None
    stored_backend = None
    stored_model = None
    query_template_hash = None
    document_prefix_hash = None

    if not exists or points_count <= 0:
        compatibility = "unknown"
    elif stored_contract is None:
        compatibility = "unknown"
        warning_codes.append("index_contract_missing")
    else:
        stored_backend = stored_contract.get("embedding_backend")
        stored_model = stored_contract.get("embedding_model")
        stored_schema_version = stored_contract.get("index_schema_version")
        query_template_hash = stored_contract.get("query_template_hash")
        document_prefix_hash = stored_contract.get("document_prefix_hash")
        if stored_backend != runtime_embedding_backend or stored_model != runtime_embedding_model:
            compatibility = "incompatible"
            blocking = True
            warning_codes.append("embedding_contract_mismatch")
        elif (
            stored_schema_version is not None
            and str(stored_schema_version) != str(settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION)
        ):
            try:
                stored_schema_number = int(str(stored_schema_version))
                expected_schema_number = int(str(settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION))
            except ValueError:
                stored_schema_number = expected_schema_number = 0
            if stored_schema_number > expected_schema_number:
                compatibility = "incompatible"
                blocking = True
                warning_codes.append("schema_version_mismatch")
            else:
                compatibility = "unknown"
                warning_codes.append("legacy_dense_only_index")
                if query_template_hash and query_template_hash != runtime_query_template_hash:
                    compatibility = "incompatible"
                    blocking = True
                    warning_codes.append("query_template_hash_mismatch")
                if document_prefix_hash and document_prefix_hash != runtime_document_prefix_hash:
                    compatibility = "incompatible"
                    blocking = True
                    warning_codes.append("document_prefix_hash_mismatch")
        elif not query_template_hash or not document_prefix_hash:
            compatibility = "unknown"
            warning_codes.append("index_contract_hash_missing")
        elif query_template_hash != runtime_query_template_hash:
            compatibility = "incompatible"
            blocking = True
            warning_codes.append("query_template_hash_mismatch")
        elif document_prefix_hash != runtime_document_prefix_hash:
            compatibility = "incompatible"
            blocking = True
            warning_codes.append("document_prefix_hash_mismatch")

    return IndexCollectionContract(
        scope=scope,  # type: ignore[arg-type]
        collection_name=store.collection_name(scope),
        exists=exists,
        points_count=points_count,
        lexical_documents=lexical_documents,
        stored_embedding_backend=stored_backend,
        stored_embedding_model=stored_model,
        stored_schema_version=stored_schema_version if stored_schema_version is not None else "unknown",
        query_template_hash=query_template_hash,
        document_prefix_hash=document_prefix_hash,
        compatibility=compatibility,  # type: ignore[arg-type]
        blocking=blocking,
        warning_codes=warning_codes,
    )


def summarize_collection_contracts(
    collection_contracts: list[IndexCollectionContract],
) -> IndexContractSummary:
    """Summarize per-collection contracts for index_status.v2."""

    warning_codes = sorted(
        {
            warning_code
            for collection_contract in collection_contracts
            for warning_code in collection_contract.warning_codes
        }
    )
    blocking = any(collection_contract.blocking for collection_contract in collection_contracts)
    if blocking:
        compatibility = "incompatible"
    elif any(collection_contract.compatibility == "unknown" for collection_contract in collection_contracts):
        compatibility = "unknown"
    else:
        compatibility = "compatible"
    incompatibility_codes = [
        code
        for code in warning_codes
        if code
        in {
            "embedding_contract_mismatch",
            "schema_version_mismatch",
            "query_template_hash_mismatch",
            "document_prefix_hash_mismatch",
        }
    ]
    return IndexContractSummary(
        compatibility=compatibility,
        blocking=blocking,
        warning_codes=warning_codes,
        incompatibility_codes=incompatibility_codes,
    )
