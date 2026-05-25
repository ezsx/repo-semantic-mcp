# P3: Agent-grade status, search shape, and exact fallback contract

## 1) Purpose

This specification covers the next implementation slice for `repo-semantic-search`:

1. Make `index_status()` agent-grade.
2. Improve semantic/hybrid search query parameters and result shape.
3. Define the exact-search fallback story.

The goal is not to add a new RAG architecture yet. The goal is to make the current MCP contract stable, machine-readable, and comfortable for agents working in large repositories.

## 2) Current findings

### 2.1 Status

Current `index_status()` already exposes repo identity, active repo, runtime/backend switch flags, timestamps, collection counts, git state, and freshness. The weak points are:

- `reason_if_unavailable` is prose, not a stable code.
- Runtime readiness, dependency readiness, and search readiness are split across `/readyz`, `/statusz`, and `index_status()`.
- Backend health/action details live in separate tools and are not summarized in status.
- `get_repo_status(repo_root)` returns different shapes for runtime repo and non-runtime repo.
- Freshness can be stale while `search_available=true`; agents need an explicit policy field.
- `index://config` manually selects fields and can drift from `IndexStatusResult`.
- Lifecycle progress has no operation id, phase, progress, or last-error metadata.

### 2.2 Search

Current search results are flat chunk rows:

- `chunk_id`
- `scope`
- `relative_path`
- `language`
- `chunk_type`
- `start_line`
- `end_line`
- `snippet`
- `symbol_path`
- `heading_path`
- `domain_tags`
- `score`
- `dense_score`
- `lexical_score`

Current filters are limited to:

- `path_prefix`
- `chunk_types`
- `domain_tags`

The weak points are:

- no `include_paths` or `exclude_paths`;
- no file extension/language filters;
- no grouped-by-file response;
- snippets are first-N normalized characters, not match-centered;
- no `why_matched` or matched terms;
- dense search filters after Qdrant retrieval, so selective filters can under-return;
- changing the response from a list to an envelope would break existing consumers.

### 2.3 Exact fallback

There is no dedicated exact-search MCP tool. Hybrid search has BM25 lexical scoring, but it is not a replacement for deterministic exact search.

The intended workflow remains:

- semantic search for conceptual discovery;
- hybrid search for concept plus identifier queries;
- local `rg` for exact identifiers, literals, env vars, route names, SQL names, errors, and paths.

For MCP-only clients, an indexed-corpus `exact_search` can be useful later, but shell-backed `rg` inside MCP should not be the default because it expands the security and performance surface.

## 3) Compatibility rules

All changes in this slice must be additive unless explicitly implemented as a new v2 tool.

Required compatibility constraints:

- Keep existing `semantic_search`, `semantic_search_code`, `semantic_search_docs`, `hybrid_search`, `hybrid_search_code`, and `hybrid_search_docs`.
- Keep returning list-like search results for existing tools.
- Do not replace `score`; add `final_score` as an alias when useful.
- Do not remove `reason_if_unavailable`; add stable codes alongside it.
- Do not make `/readyz` require `search_available`.
- Do not make git freshness stale block search by default.
- Keep the current empty-result compatibility behavior for existing search tools until FastMCP empty list handling is replaced intentionally. Today an empty search result may serialize as the string `"[]"`.
- Do not make lifecycle tools run implicitly from status or search.

## 4) Status contract v2

### 4.1 Additive top-level fields

Add these fields to `IndexStatusResult`:

```python
contract_version: str = "index_status.v2"
status_kind: str = "runtime_repo_status"
availability: StatusAvailability
runtime: RuntimeStatusSummary
backend: BackendStatusSummary
index_contract: IndexContractStatus
counts: IndexCounts
lifecycle: LifecycleStatus
watcher: WatcherStatusSummary
warnings: list[StatusWarning]
```

Keep current top-level fields for backward compatibility.

### 4.1.1 Pre-ready status boundary

`index_status()` is an MCP tool and currently requires a configured and ready runtime. It does not have to report `runtime_ready=false`.

Pre-ready and bootstrap-failure diagnostics belong to HTTP `/statusz` and `/readyz`:

- `/statusz` must remain callable during bootstrap and after bootstrap failure.
- `/statusz` may include `index_status` only when runtime status can be collected safely.
- `index_status.availability.runtime_ready` means "the MCP runtime accepted this tool call", not "bootstrap may still be pending".

If later we want MCP-level pre-ready status, add a separate tool such as `transport_status()` instead of weakening every search/status tool gate.

### 4.2 Availability

```python
class StatusAvailability(BaseModel):
    runtime_ready: bool
    dependencies_ready: bool | None = None
    search_available: bool
    search_unavailable_codes: list[str] = []
    host_action_required: bool = False
    next_actions: list[StatusAction] = []
```

Stable unavailable codes:

- `repo_not_registered`
- `runtime_repo_not_active`
- `index_missing`
- `lifecycle_stale_blocking`
- `indexing_in_progress`
- `index_error`
- `repo_disabled`
- `backend_switch_required`
- `backend_unhealthy`
- `embedding_contract_mismatch`
- `index_empty`
- `dependency_unavailable`
- `unknown`

Stale terminology:

- `freshness_stale` means the git worktree or HEAD differs from the indexed revision. This is nonblocking by default.
- `lifecycle_stale_blocking` means the registry/index lifecycle state is `stale`, usually because indexed config, include/doc rules, or embedding contract changed. This blocks search under the current P2 lifecycle contract.
- `embedding_contract_mismatch` is always blocking.

The existing `reason_if_unavailable` should become a human-readable explanation derived from the first code.

### 4.3 Actions

```python
class StatusAction(BaseModel):
    code: str
    severity: Literal["info", "warning", "blocking"]
    title: str
    detail: str | None = None
    tool_hint: str | None = None
    host_command_hint: str | None = None
```

Examples:

- `build_index_required`
- `runtime_switch_required`
- `backend_switch_required`
- `explicit_rebuild_recommended`
- `use_exact_search_fallback`
- `check_backend_action_plan`

Actions must never execute lifecycle work. They only describe the next explicit action.

```python
class StatusWarning(BaseModel):
    code: str
    severity: Literal["info", "warning"]
    detail: str | None = None
```

### 4.4 Runtime summary

```python
class RuntimeStatusSummary(BaseModel):
    bootstrap_phase: str | None = None
    bootstrap_error: str | None = None
    mounted_repo_root: str
    logical_repo_root: str
    active_repo_root: str | None = None
    runtime_switch_required: bool
    runtime_switch_command_hint: str | None = None
```

Implementation note: `SearchService.index_status()` does not currently know `_BOOTSTRAP_STATE`. Either pass a lightweight status provider into `SearchService`, or let `mcp_server.index_status()` enrich the model before dumping.

### 4.5 Backend summary

```python
class BackendStatusSummary(BaseModel):
    runtime_embedding_backend_id: str | None = None
    selected_embedding_backend_id: str | None = None
    backend_type: str | None = None
    model_name: str | None = None
    compatible: bool
    switch_required: bool
    healthy: bool | None = None
    health_error: str | None = None
    managed_by_service: bool | None = None
    autostart_policy: str | None = None
    location_type: str | None = None
    device_type: str | None = None
    action_plan_tool: str = "get_backend_action_plan"
```

The status path should avoid expensive backend health checks by default. If health is not cheap or not available, return `healthy=None` and point to `get_backend_action_plan`.

### 4.6 Index contract

```python
class IndexContractStatus(BaseModel):
    profile: str
    embedding_backend: str
    embedding_model: str
    schema_version: int
    query_template_hash: str | None = None
    document_prefix_hash: str | None = None
    compatibility: Literal["compatible", "incompatible", "unknown"]
    blocking: bool
    incompatibility_codes: list[str] = []
    collections: list[IndexCollectionContract] = []
```

Collection contract should include:

- `scope`
- `collection_name`
- `exists`
- `points_count`
- `lexical_documents`
- `stored_embedding_backend`
- `stored_embedding_model`
- `stored_schema_version`
- `compatibility`
- `blocking`

```python
class IndexCollectionContract(BaseModel):
    scope: ChunkScope
    collection_name: str
    exists: bool
    points_count: int
    lexical_documents: int | None = None
    stored_embedding_backend: str | None = None
    stored_embedding_model: str | None = None
    stored_schema_version: int | Literal["unknown"] | None = None
    query_template_hash: str | None = None
    document_prefix_hash: str | None = None
    compatibility: Literal["compatible", "incompatible", "unknown"]
    blocking: bool
    warning_codes: list[str] = []
```

Hash and migration rules:

- hash algorithm: SHA-256 over the exact UTF-8 string value, stored as lowercase hex;
- current runtime hashes are computed from `SEMANTIC_MCP_QUERY_TEMPLATE` and `SEMANTIC_MCP_DOCUMENT_PREFIX`;
- new indexes should store `query_template_hash`, `document_prefix_hash`, and `index_schema_version` in chunk payloads;
- old indexes that lack template/prefix hashes should report `compatibility="unknown"`, `blocking=false`, and add warning code `index_contract_hash_missing`;
- missing template/prefix hashes are nonblocking in the first migration slice;
- backend/model mismatch and schema-version mismatch report `compatibility="incompatible"` and `blocking=true`;
- stored schema version should be read from payload when available. If absent, report `unknown` and warn, but do not block old indexes solely for missing schema metadata.

### 4.7 Counts

```python
class IndexCounts(BaseModel):
    code_points: int
    docs_points: int
    total_points: int
    code_lexical_documents: int | None = None
    docs_lexical_documents: int | None = None
```

This duplicates `collections[]` intentionally for agent ergonomics.

Lexical document counts must not force full corpus scroll/tokenization during status. Prefer persisted registry counts. If exact lexical counts are not already cached cheaply, return `None` and add warning code `lexical_counts_not_loaded`.

### 4.8 Lifecycle

```python
class LifecycleStatus(BaseModel):
    state: RepoStatus
    action_in_progress: bool
    operation_id: str | None = None
    phase: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    progress: dict[str, int | float | str] = {}
    last_error: str | None = None
    last_error_at: str | None = None
```

Initial implementation can set only `state`, `action_in_progress`, `updated_at`, and `last_error`. Operation progress can come later.

### 4.9 Freshness policy

Extend current `freshness`:

```python
class FreshnessPolicy(BaseModel):
    search_allowed_when_stale: bool = True
    stale_severity: Literal["none", "info", "warning", "blocking"]
    requires_rebuild: bool = False
    exact_fallback_recommended: bool = False
```

Rules:

- changed/untracked indexable files make `stale_severity="warning"`;
- embedding contract mismatch makes `stale_severity="blocking"`;
- unknown indexed commit makes `stale_severity="info"`;
- git freshness stale must not block search unless config later explicitly opts into strict mode;
- lifecycle stale remains blocking under the current lifecycle contract.

### 4.10 Watcher

```python
class WatcherStatusSummary(BaseModel):
    enabled: bool
    running: bool
    policy: str = "single_active_repo"
    last_event_at: str | None = None
    last_error: str | None = None
    last_incremental_update_ts: str | None = None
```

Initial implementation can map current watcher fields and timestamps only.

### 4.11 Repo status v2

Keep `get_repo_status(repo_root)` backward-compatible for now.

Add a new tool later if we need unified shape:

```python
get_repo_status_v2(repo_root: str | None = None) -> RepoStatusEnvelope
```

```python
class RepoStatusEnvelope(BaseModel):
    contract_version: str = "repo_status.v2"
    status_kind: Literal["runtime_repo_status", "registered_repo_status"]
    repo_root: str
    runtime_visible: bool
    runtime_status: IndexStatusResult | None = None
    registry_entry: RepoRegistryEntry | None = None
    unavailable_fields: list[str] = []
```

For non-runtime registered repos, `runtime_status=None`, `registry_entry` is populated, and fields requiring mounted filesystem, Qdrant collection access, or backend runtime health are listed in `unavailable_fields`.

## 5) Search contract v1.1

### 5.1 Additive query parameters

Add optional parameters to semantic and hybrid tools:

```python
include_paths: list[str] | None = None
exclude_paths: list[str] | None = None
file_extensions: list[str] | None = None
languages: list[str] | None = None
max_results_per_file: int | None = None
snippet_mode: Literal["chunk_start", "query_centered"] = "chunk_start"
include_explanations: bool = True
```

Keep `path_prefix` as a compatibility alias. Internally normalize it to the boundary-aware prefix predicate defined below, not to naive string startswith.

All search tools must enforce caps:

- max query length: 1000 chars;
- max `top_k`: 100;
- max filter values per family: 50;
- max glob length: 240 chars;
- max returned snippet chars per result: 500;
- max line matches per result: 5;
- max explanation strings per result: 8.

### 5.2 Search filters

Introduce an internal `SearchFilters` model:

```python
class SearchFilters(BaseModel):
    path_prefix: str | None = None
    include_paths: list[str] = []
    exclude_paths: list[str] = []
    file_extensions: list[str] = []
    languages: list[str] = []
    chunk_types: list[str] = []
    domain_tags: list[str] = []
```

Semantics:

- include paths are glob-like repo-relative patterns;
- exclude paths are glob-like repo-relative patterns;
- file extensions normalize to lower-case with leading dot;
- languages match `SearchResult.language`;
- domain tags keep current intersection behavior;
- all active filter families are ANDed;
- within a family, multiple values are ORed.

Path normalization and safety:

- normalize `\` to `/`;
- reject absolute paths, drive-qualified paths, empty path segments, and `..` segments;
- matching is case-sensitive and uses repo-relative POSIX-style paths;
- include filters run before exclude filters;
- exclude filters always win;
- `path_prefix="src"` matches `src` and `src/...`, not `src2/...`;
- glob semantics should use `fnmatchcase` over normalized repo-relative paths;
- a prefix alias should be normalized to a boundary-aware predicate, not naive string startswith.

### 5.3 Dense search retrieval under filters

Current dense search can under-return because it fetches an unfiltered Qdrant shortlist first. The implementation should improve this in a staged approach:

1. v1.1: use an expanding candidate limit when filters are active, for example 80, 160, 320, until either `top_k` final kept results are found after dedupe, cross-scope merge, and `max_results_per_file`, or a configured max candidate cap is reached.
2. v1.1: if the cap is reached before enough final kept results are found, return available results. Existing list-shaped tools cannot carry response-level warnings on empty results, so this best-effort state is documented but not represented in the old response shape.
3. v1.2: add Qdrant payload filters and payload indexes for fields that can be filtered natively.

Do not claim complete filtered recall for dense-only semantic search until Qdrant-side filtering or a full-corpus fallback exists. Hybrid lexical candidates are full-corpus over indexed chunks after lexical cache load, so hybrid can provide stronger recall for exact terms.

### 5.4 Additive result fields

Extend `SearchResult` with optional fields:

```python
repo_root: str | None = None
line_range: str | None = None
file_extension: str | None = None
final_score: float | None = None
match_type: Literal["semantic", "hybrid", "lexical", "exact"] | None = None
matched_terms: list[str] = []
why_matched: list[str] = []
snippet_start_line: int | None = None
snippet_end_line: int | None = None
line_matches: list[LineMatch] = []
```

Keep existing fields. `line_range` is a convenience string like `12-28`; `start_line` and `end_line` remain authoritative.

Hybrid implementation requirement:

- result shaping that needs line matches or query-centered snippets must operate on full chunk text;
- do not reconstruct dense-side hybrid chunks from `SearchResult.snippet`;
- either keep the original `ChunkRecord` through hybrid ranking or refetch the chunk before final shaping.

### 5.5 Line matches

```python
class LineMatch(BaseModel):
    line: int
    text: str
    matched_terms: list[str] = []
```

For semantic-only results, `line_matches` can be empty. For hybrid/exact-like matches, fill it from lexical terms when cheap.

### 5.6 Snippets

Modes:

- `chunk_start`: current behavior, kept as default for compatibility.
- `query_centered`: prefer lines containing lexical query tokens; otherwise fall back to chunk start.

Limits:

- max snippet chars: 500 by default;
- max line matches per result: 5;
- never return full chunk text from search tools.

If full chunk text is unavailable or line metadata is approximate, return a normal snippet and add `why_matched` entry `line_matches_unavailable`.

### 5.6.1 Max results per file

`max_results_per_file` applies after scoring, dedupe, and cross-scope merge.

`max_results_per_file=None` means no per-file cap.

Algorithm:

1. Rank all candidate chunks globally by final score.
2. Walk ranked candidates in order.
3. Keep a candidate if that file has fewer than `max_results_per_file` already kept.
4. Stop when `top_k` kept results are available or candidates are exhausted.

This preserves score ordering while preventing a single file from dominating the final list.

### 5.7 Grouped results

Do not change existing tools to return an envelope.

Add a new optional tool after v1.1 fields are stable:

```python
search_grouped(
    query: str,
    mode: Literal["semantic", "hybrid"] = "hybrid",
    top_k_files: int = 10,
    max_chunks_per_file: int = 3,
    ...
) -> SearchGroupedResponse
```

Response:

```python
class SearchGroupedResponse(BaseModel):
    query: str
    mode: str
    files: list[FileSearchGroup]
```

```python
class FileSearchGroup(BaseModel):
    relative_path: str
    scope: ChunkScope
    language: str
    file_extension: str
    best_score: float
    match_count: int
    why_file_matched: list[str] = []
    chunks: list[SearchResult]
```

This is a separate tool because existing consumers expect list-shaped results.

### 5.8 Empty results

Existing tools must preserve `_search_payload()` behavior:

- non-empty results return list-like row payloads;
- empty results may return the JSON string `"[]"` for FastMCP compatibility.

Tests must cover this explicitly before changing the behavior.

### 5.9 Search response with metadata

Existing list-shaped tools cannot carry response-level metadata, especially when results are empty. Add a new tool for agent-grade diagnostics:

```python
search_v2(
    query: str,
    mode: Literal["semantic", "hybrid"] = "hybrid",
    ...
) -> SearchResponse
```

```python
class SearchResponse(BaseModel):
    contract_version: str = "search.v2"
    query: str
    mode: str
    results: list[SearchResult]
    warnings: list[SearchWarning] = []
    diagnostics: SearchDiagnostics | None = None
```

```python
class SearchWarning(BaseModel):
    code: str
    severity: Literal["info", "warning"]
    detail: str | None = None
```

```python
class SearchDiagnostics(BaseModel):
    candidate_limit: int | None = None
    candidates_scanned: int | None = None
    filtered_candidates: int | None = None
    final_results: int
    dense_best_effort: bool = False
    lexical_cache_used: bool = False
```

Use `search_v2` when callers need response-level warnings such as:

- `filtered_dense_search_best_effort`;
- `empty_result_best_effort`;
- `stale_index_verify_with_exact_search`;
- `lexical_cache_loaded`.

Do not retrofit these warnings into old list-shaped tools with sentinel rows.

## 6) Exact fallback contract

### 6.1 Agent policy

Exact search priority:

1. Use local `rg` when the agent has filesystem access and the query is exact.
2. Use `hybrid_search` for mixed semantic plus identifier queries.
3. Use `semantic_search` for conceptual discovery.
4. Verify semantic/hybrid findings by reading files or running exact search before editing.

Exact query examples:

- env vars;
- route names;
- SQL table/column names;
- error messages;
- filenames and paths;
- quoted literals;
- function/class names when exact spelling is known.

Safe `rg` guidance:

- run from the target repo root only;
- do not interpolate untrusted user text through a shell command string;
- use argument arrays or equivalent safe invocation when available;
- inherit or mirror the MCP index exclude policy for common secret/binary paths;
- include excludes for `.git/**`, `.env*`, key/cert files, archives, binary media, and local artifact directories;
- keep output bounded with context and match limits.

### 6.2 Status guidance

Add to `index_status.availability.next_actions` when appropriate:

- `use_exact_search_fallback` only for repo/index conditions, for example when git freshness is stale and exact verification is recommended;
- `exact_fallback_authority="local_rg"` in status or config resource.

This is guidance only. The MCP must not run lifecycle or shell commands implicitly.

Query-style guidance does not belong in `index_status()` because status has no query input. Put query-style guidance in:

- MCP server instructions;
- search tool docstrings;
- future query-time response metadata;
- project agent policy docs.

### 6.3 Optional MCP-only exact search

Add later, not in the first implementation pass unless review approves:

```python
exact_search(
    query: str,
    top_k: int = 20,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    case_sensitive: bool | None = None,
    whole_word: bool = False,
    max_matches_per_file: int = 5,
) -> list[SearchResult]
```

Scope:

- search only indexed chunk payloads;
- no shell execution;
- no regex initially;
- include stale-corpus warning in `why_matched` when returning list-shaped results;
- if response-level freshness metadata is required, add an envelope tool instead of overloading existing list-shaped tools;
- return `match_type="exact"`.

Limits:

- cap query length;
- cap `top_k`;
- cap matches per file;
- cap snippet bytes.

Case behavior:

- `case_sensitive=None` means smart-case: lowercase queries are case-insensitive, mixed/uppercase queries are case-sensitive;
- `case_sensitive=True` forces case-sensitive;
- `case_sensitive=False` forces case-insensitive;
- `whole_word` must use identifier-aware boundaries, not only whitespace.

## 7) Implementation slices

### Slice A: Status v2 additive models

Files:

- `services/repo_semantic/models.py`
- `services/repo_semantic/search_service.py`
- `services/repo_semantic/mcp_server.py`
- `tests/test_index_status_contract.py`

Deliverables:

- stable unavailable codes;
- `availability`, `runtime`, `backend`, `index_contract`, `counts`, `lifecycle`, `watcher`, and `warnings`;
- backward-compatible existing fields;
- tests for registered, missing index, backend switch, stale freshness, and active repo mismatch.

### Slice B: Search filters and additive result fields

Files:

- `services/repo_semantic/models.py`
- `services/repo_semantic/search_service.py`
- `services/repo_semantic/mcp_server.py`
- `tests/`

Deliverables:

- `SearchFilters` internal model;
- include/exclude path filters;
- file extension and language filters;
- optional `max_results_per_file`;
- additive result fields;
- query-centered snippet mode;
- `search_v2` envelope for diagnostics and response-level warnings;
- tests for filter semantics and compatibility with existing result fields.

### Slice C: Exact fallback guidance

Files:

- `docs/`
- `services/repo_semantic/models.py`
- `services/repo_semantic/search_service.py`

Deliverables:

- exact fallback guidance in status/config;
- safe local `rg` guidance with repo-root and exclude-policy constraints;
- no shell-backed exact search;
- optional `exact_search` left behind a review decision.

### Slice D: Optional grouped search

This should be implemented after additive result fields are stable.

Deliverables:

- new `search_grouped` tool;
- no change to existing search tool response shape.

## 8) Test strategy

Unit tests should not require Qdrant or embedding backend.

Use fake stores/providers for:

- status shape;
- availability codes;
- stale policy;
- search filter matching;
- snippet generation;
- max results per file;
- exact indexed-corpus matching if implemented.

Add integration/smoke tests later for:

- real MCP tool call shape;
- empty-result compatibility;
- benchmark script compatibility;
- `/statusz` payload includes v2 fields.

## 9) Open review questions

1. Should `exact_search` be part of this implementation slice, or should we ship guidance first?
2. Should strict git freshness mode exist as an opt-in setting, or should freshness stale always remain warning-only?
3. Should backend health be checked inside `index_status()`, or should `index_status()` only point to `get_backend_action_plan`?
4. Should grouped search be a new tool only, or also an optional `group_by_file=true` mode later?
5. Should Qdrant payload filters be part of the first search-filter implementation, or a later performance slice after best-effort filtered dense search lands?

## 10) Acceptance criteria

This slice is complete when:

1. `index_status()` has stable machine-readable readiness and unavailable codes.
2. Agents can determine next action without parsing prose.
3. Existing search tools remain backward-compatible.
4. Search supports include/exclude paths and file extension/language filters with documented best-effort dense recall until Qdrant filters land.
5. Search results include line/rationale fields without changing old fields.
6. `search_v2` carries response-level warnings without changing old list-shaped tools.
7. Empty search result compatibility is preserved or intentionally replaced with tests.
8. Exact search guidance is explicit, repo-root-bounded, mirrors exclude policy, and says when to use local `rg`.
9. Tests cover the new contracts without external services.
