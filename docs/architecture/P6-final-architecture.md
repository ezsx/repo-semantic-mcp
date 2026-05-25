# P6: Final architecture for `repo-semantic-search`

Status: accepted architecture baseline for the next specification phase.

Source: GPT-5.5 Pro architecture pass over P3, P4, P5, and P6 artifacts.

Note: the external HippoRAG deep-research report was not uploaded directly into
that pass. GPT-5.5 Pro used
`docs/research/hipporag/P5-hipporag-research-analysis.md` as the accepted
internal analysis of that report.

## 1) Final architecture verdict

`repo-semantic-search` should become:

```text
agent query / agent-provided subqueries
  -> dense raw retrieval over Qdrant
  -> sparse code-normalized retrieval over Qdrant
  -> weighted RRF seed fusion
  -> optional deterministic typed graph expansion from strong seeds
  -> weighted RRF over dense+sparse+graph branches
  -> optional explicit rerank / optional explicit ColBERT later
  -> grounded repository context envelope
```

This is not a GraphRAG summarizer, not a hidden coding agent, and not a
final-answer system. The MCP remains a retrieval substrate that gives frontier
coding agents grounded context, diagnostics, freshness state, and next-step
hints. The planner and editor remain outside the MCP.

The non-negotiable baseline is server-side dense+sparse retrieval with rank
fusion. The current local BM25 scroll-cache and direct dense/BM25 score blending
are transitional implementation details, not future architecture.

The graph layer should be HippoRAG-style only in the narrow sense of seeded graph
propagation. It should not use OpenIE triples as the core code index, and it
should not replace dense+sparse retrieval. Dense raw retrieval plus sparse
code-normalized retrieval produce seeds; graph expansion is an optional branch;
results stay grounded by file/chunk/line evidence.

Qdrant is the vector substrate for this route because it supports dense+sparse
hybrid queries with RRF-style fusion and named dense/sparse vectors, and it also
supports multivector representations for later ColBERT-style late interaction.

## 2) Tool-surface decision

Add a new primary tool:

```text
repo_context_search
```

Do not call it `search_v3` as the main public name.

Reason: `search_v3` implies an incremental replacement of `search_v2`. The new
tool is not just another search version; it is an agent-facing context retrieval
route with multi-query merge, branch diagnostics, optional graph expansion,
evidence paths, coverage diagnostics, and recommended next actions.

Existing list-shaped search tools and `search_v2` remain stable for
compatibility.

Keep:

```text
semantic_search
semantic_search_code
semantic_search_docs
hybrid_search
hybrid_search_code
hybrid_search_docs
search_v2
search_grouped       # optional / later
```

Add:

```text
repo_context_search
graph_status
build_graph          # explicit lifecycle operation
rebuild_graph        # explicit lifecycle operation
update_graph         # optional explicit incremental operation
```

Defer:

```text
indexed_exact_search
```

If implemented, name it `indexed_exact_search`, not just `exact_search`, unless
docs and response shape make it impossible to confuse with authoritative
filesystem `rg`. Local `rg` remains the preferred exact fallback when the coding
agent has filesystem access, and shell-backed `rg` inside MCP should not be the
default.

## 3) Compatibility boundary

The P3 compatibility contract remains intact:

- Existing list-shaped tools keep returning list-like results.
- Do not retrofit route-level warnings into legacy tools using sentinel rows.
- Keep `score`; add `final_score` only as an alias/extension.
- Preserve empty-result compatibility behavior until FastMCP handling is
  intentionally changed and tested.
- Lifecycle operations must not run implicitly from status or search.

Architecture layers:

```text
Legacy search tools
  -> backward-compatible rows
  -> may internally benefit from new retrieval implementation
  -> no graph branch by default

search_v2
  -> stable envelope from P3
  -> dense/sparse hybrid, diagnostics, warnings
  -> no mandatory graph semantics

repo_context_search
  -> new context envelope
  -> dense+sparse+optional graph
  -> branch diagnostics, evidence paths, coverage, next actions
```

## 4) Storage architecture

### 4.1 Qdrant chunk retrieval index

Keep separate logical `code` and `docs` scopes. Do not collapse them into one
physical collection unless a later implementation review proves it simplifies
filtering and improves performance. The current split is already part of the
operational baseline and helps agents reason about scope.

Each collection should evolve to named vectors:

```text
dense vector:
  name: dense
  distance: cosine
  contract:
    embedding_backend_id
    embedding_model
    embedding_dim
    query_template_hash
    document_prefix_hash

sparse vector:
  name: sparse
  contract:
    lexical_analyzer_version
    sparse_encoder_kind
    vocabulary_hash or hashing_scheme_id
    bm25_params / sparse_weighting_params
    corpus_stats_hash
```

Do not use neural sparse models as the first sparse implementation unless there
is a strong operational reason. A deterministic code-aware BM25-style sparse
encoder is the better first target: inspectable, cheap, exact-anchor friendly,
and free of another model-serving dependency before the route is stable.

Persist these payload fields in addition to current metadata:

```text
file_extension
content_hash
index_schema_version
dense_contract_id
sparse_contract_id
lexical_analyzer_version
is_indexable
is_generated
path_segments
path_prefixes
```

`path_segments` and `path_prefixes` help because Qdrant-side filtering on
arbitrary globs is not enough. Prefix-style filters should be pushed down where
possible; full glob semantics can remain client-side with explicit best-effort
diagnostics.

Payload indexes:

```text
scope
relative_path
file_extension
language
chunk_type
domain_tags
is_generated
content_hash
path_prefixes
```

### 4.2 Sparse index design

Do not keep the current "scroll all chunks into local BM25 cache at query time"
design as a production retrieval path.

Preferred sparse implementation:

```text
persisted vocabulary + BM25-style sparse vectors
token -> stable integer id
corpus stats persisted in sparse manifest
query vector built from same lexical analyzer and vocabulary
```

Acceptable fallback:

```text
stable hashed sparse indices
collision-aware diagnostics
explicit hash scheme id
```

Preferred is better. Hashing removes a vocabulary table but weakens exact-anchor
explainability because collisions can create surprising matches. For code
retrieval, explainability matters more than avoiding a small manifest.

Sparse compatibility should be visible in status:

```text
sparse_available: bool
sparse_contract_compatible: bool
sparse_unavailable_codes: [...]
sparse_vector_points: int
sparse_manifest_hash: str | None
```

Old indexes without sparse vectors should not silently pretend to have full
hybrid retrieval. During one compatibility window, legacy `hybrid_search` may
use the local BM25 cache with a warning in `search_v2` diagnostics, but
`repo_context_search` should report `sparse_branch_unavailable` and recommend an
explicit rebuild.

### 4.3 SQLite graph artifact

Use SQLite for graph v1. Do not introduce Neo4j, DuckDB, or a graph database in
the first graph implementation.

Use one graph artifact per repo/profile, not split by `code` and `docs`.

Reason: the graph's value is precisely in crossing code/docs/tests/config/
deploy/policy boundaries. Splitting the graph by scope would recreate the
cross-file inference burden the MCP is supposed to reduce.

Recommended SQLite tables:

```text
graph_metadata(
  repo_id,
  repo_root,
  profile,
  schema_version,
  built_commit,
  built_at,
  include_rules_hash,
  exclude_rules_hash,
  extractor_versions_json,
  source_index_contract_json,
  graph_contract_hash
)

graph_files(
  relative_path,
  content_hash,
  mtime,
  graph_state,              -- ready | stale | partial | error | excluded
  indexed_at,
  last_extracted_at,
  error_code,
  error_detail
)

graph_nodes(
  node_id,
  node_type,
  key,
  relative_path,
  chunk_id,
  start_line,
  end_line,
  payload_json
)

graph_node_terms(
  node_id,
  term,
  normalized_term,
  term_type,                -- symbol | env_var | path | route | config_key | doc_ref | ...
  case_sensitive,
  source
)

graph_node_chunks(
  node_id,
  chunk_id,
  relation,                 -- primary | containing | mention | nearest
  weight
)

graph_edges(
  edge_id,
  source_node_id,
  target_node_id,
  edge_type,
  confidence,
  extractor,
  relative_path,
  payload_json
)
```

Indexes:

```text
graph_nodes(node_type, key)
graph_nodes(relative_path)
graph_nodes(chunk_id)
graph_node_terms(normalized_term, term_type)
graph_node_chunks(chunk_id)
graph_edges(source_node_id, edge_type)
graph_edges(target_node_id, edge_type)
graph_edges(relative_path)
graph_files(relative_path, content_hash)
```

`graph_node_terms` and `graph_node_chunks` are part of the first graph schema.
Without them, seed binding becomes ad hoc and graph candidates that map to files
or symbols spanning multiple chunks become awkward.

## 5) Indexing and lifecycle architecture

Indexing pipeline:

```text
discover files
  -> apply include/exclude policy
  -> chunk files
  -> extract metadata
  -> build dense embedding
  -> build sparse lexical vector
  -> upsert Qdrant points
  -> persist index contract
  -> optionally build/update graph artifact through explicit lifecycle step
```

Graph build must be explicit.

Initial P5 behavior:

```text
build_index(...)
  -> builds Qdrant dense+sparse index only

build_graph(...)
  -> builds graph from indexed chunks + source files

rebuild_graph(...)
  -> drops/rebuilds graph artifact

update_graph(paths=[...])
  -> optional explicit incremental graph update
```

Later, after graph is stable:

```text
build_index(include_graph="profile_default" | true | false)
```

This remains safe because `build_index` is an explicit lifecycle operation. What
must never happen is `search`, `repo_context_search`, `index_status`, or
`graph_status` deciding to rebuild or update graph/index state automatically.

Graph freshness is independent from vector freshness. A vector index may be
usable while the graph is missing or stale. In that case, dense+sparse search
continues, graph expansion is skipped or marked unavailable, and the response
carries a warning.

## 6) Retrieval route

### 6.1 Request contract

Recommended request shape:

```python
class RepoContextSearchRequest(BaseModel):
    query: str
    subqueries: list[str] = []

    route: Literal[
        "auto",
        "semantic",
        "hybrid",
        "exact_handoff",
    ] = "auto"

    graph_mode: Literal[
        "off",
        "auto",
        "expand",
    ] = "auto"

    scope: Literal["all", "code", "docs"] = "all"

    include_paths: list[str] = []
    exclude_paths: list[str] = []
    file_extensions: list[str] = []
    languages: list[str] = []
    chunk_types: list[str] = []
    domain_tags: list[str] = []

    top_k: int = 20
    max_results_per_file: int | None = 3
    snippet_mode: Literal["chunk_start", "query_centered"] = "query_centered"

    include_evidence_paths: bool = True
    include_diagnostics: bool = True

    rerank_mode: Literal["off", "auto", "on"] = "off"
```

Hard caps should follow P3: query length, `top_k`, filter values, glob length,
returned snippet chars, line matches, and explanation count must remain bounded.

### 6.2 Query preprocessing

The MCP may do deterministic preprocessing. It must not run an internal LLM
planner.

For every request:

```text
queries_used = [original_query] + subqueries
```

The original query is always included and always first. This prevents subqueries
or rewrites from dropping exact anchors.

Preprocessing should produce:

```text
raw_dense_text
lexical_token_bag
exact_anchor_candidates
query_intent_signals
filter_plan
```

Dense text remains raw. Do not normalize, split, lowercase, or rewrite it.

Sparse text uses code-aware lexical normalization:

```text
preserve exact token:
  DATABASE_URL
  /api/v1/users/{id}
  UserRepository.findByEmail
  create_users_table
  "permission denied"

also add normalized expansions:
  database url
  api v1 users id
  user repository find by email
  create users table
  permission denied
```

Normalization handles:

```text
snake_case
kebab-case
camelCase
PascalCase
dotted.names
slash/route/URL paths
SQL identifiers
env vars
CLI flags
file path fragments
quoted literals
error strings
```

Do not use PRF, HyDE, or LLM query rewriting by default. In code search, these
often erase identifiers and literals.

### 6.3 Dense branch

Dense branch:

```text
input: raw query
vector: dense
candidate limit:
  dense_limit = min(max_dense_cap, max(50, top_k * 8))
filters:
  Qdrant payload filters where possible
  client-side glob filters where necessary
output:
  ranked dense candidates with dense rank and dense score
```

For `route="semantic"`, this is the only retrieval branch.

### 6.4 Sparse branch

Sparse branch:

```text
input: code-normalized lexical token bag
vector: sparse
candidate limit:
  sparse_limit = min(max_sparse_cap, max(100, top_k * 12))
filters:
  same filter plan
output:
  ranked sparse candidates with sparse rank, sparse score, matched terms
```

For code repositories, sparse recall should generally be wider than dense recall.
Exact anchors, env vars, routes, table names, config keys, file names, and error
strings should be found here without Python-side full-corpus scrolling.

### 6.5 Dense+sparse seed fusion

Use service-level weighted RRF, not raw score blending.

Recommended initial policy:

```text
rrf_k = 60

weights:
  dense: 1.0
  sparse: 1.2
```

For conceptual queries with weak lexical anchors:

```text
dense: 1.1
sparse: 1.0
```

For exact-anchor-heavy queries:

```text
dense: 0.8
sparse: 1.4
```

These are starting policies, not magic constants. The critical point is that
dense scores, sparse scores, graph path scores, and reranker scores must not be
blended as if they were calibrated probabilities.

### 6.6 Graph trigger policy

`graph_mode` behavior:

```text
off:
  never expand graph

expand:
  expand graph if graph is available and compatible;
  return warning if graph is missing/stale/incompatible

auto:
  expand only when deterministic triggers say graph is likely useful
```

`auto` should enable graph expansion for:

```text
cross-file flow queries:
  "where is this flow implemented"
  "find handler, tests, docs"
  "how does config reach runtime"
  "where is this env var used"
  "what migration/schema owns this table"
  "which deploy contract controls this"

relationship queries:
  "where used"
  "tests for"
  "docs for"
  "config for"
  "policy for"
  "handler for"
  "caller/callee"       # only after call edges exist

infrastructure anchors:
  env vars
  route literals
  config keys
  SQL table/column names
  migration names
  package/deploy identifiers
  CLI flags
```

`auto` should suppress graph expansion for:

```text
plain file-path lookup
plain exact symbol lookup
plain quoted literal lookup
single high-confidence sparse hit with no relationship intent
top_k very small and no graph-related intent
graph stale around the seed paths
```

Exact anchors often suppress graph expansion for simple literal lookup, but
infrastructure anchors such as env vars, routes, SQL tables, and error strings
can be exactly where narrow graph expansion helps.

### 6.7 Seed binding

Use only strong seeds for graph expansion.

Recommended defaults:

```text
max_graph_seed_chunks = 20
min_seed_rank_source = top 50 dense/sparse fused candidates
max_hops_v1 = 1
max_graph_candidates = 200
```

Bind seeds to graph nodes by:

```text
chunk_id -> chunk node
relative_path -> file node
symbol_path -> symbol node
heading_path -> doc_section node
matched terms -> graph_node_terms
env var anchors -> env_var nodes
config key anchors -> config_item nodes
```

If a seed cannot bind to any graph node, it remains a dense/sparse result. Do not
fabricate graph explanations.

### 6.8 Graph expansion v1

Initial node types:

```text
file
chunk
symbol
doc_section
config_item
env_var
policy_scope
```

Initial edge types:

```text
file_contains_chunk
chunk_defines_symbol
file_contains_doc_section
doc_references_symbol
doc_references_config
config_defines_env
code_reads_env
policy_applies_to_path
chunk_mentions_path_or_symbol
```

Do not include full call graphs, multi-language import graphs, framework route
extraction, SQL lineage, or test-target inference in the first graph slice.
Those are high-value later edges, but adding them before graph freshness, seed
binding, evidence paths, and diagnostics are stable will make debugging much
harder.

Graph path scoring v1 should be deterministic:

```text
path_score =
    seed_rank_prior
  + exact_anchor_bonus
  + edge_type_weight
  + multi_seed_support_bonus
  - hop_penalty
  - high_degree_penalty
  - generated_or_vendor_penalty
  - stale_edge_penalty
```

Then convert graph candidates to ranks and fuse by RRF. Do not blend raw graph
path scores directly with dense or sparse scores.

### 6.9 Final fusion and diversity

Final fusion:

```text
branches:
  dense
  sparse
  graph

fusion:
  weighted RRF

default weights:
  dense: 1.0
  sparse: 1.2
  graph: 0.7 in graph_mode=auto
  graph: 1.0 in graph_mode=expand
```

After fusion:

```text
dedupe by chunk_id
apply max_results_per_file
prefer non-generated files
preserve global rank order
produce file_groups summary
shape snippets and line matches
attach diagnostics
```

`max_results_per_file` should happen after scoring and dedupe.

## 7) `repo_context_search` response contract

Recommended response:

```python
class RepoContextSearchResponse(BaseModel):
    contract_version: Literal["repo_context_search.v1"]
    query: str
    queries_used: list[QueryUsage]

    route_requested: str
    route_used: str

    graph_mode_requested: str
    graph_mode_effective: str

    results: list[RepoContextResult]
    file_groups: list[FileContextGroup] = []

    warnings: list[SearchWarning] = []
    diagnostics: RepoContextDiagnostics

    recommended_next_actions: list[RecommendedAction] = []
```

Result object extends P3 `SearchResult`:

```python
class RepoContextResult(SearchResult):
    final_rank: int
    final_score: float

    origin_branches: list[Literal[
        "dense",
        "sparse",
        "graph",
        "indexed_exact",
    ]] = []

    origin_queries: list[str] = []

    branch_ranks: dict[str, int] = {}
    branch_scores: dict[str, float] = {}

    matched_terms: list[str] = []
    uncovered_terms: list[str] = []

    evidence_paths: list[EvidencePath] = []

    freshness: ResultFreshness
    verification: VerificationHint
```

Evidence path:

```python
class EvidencePath(BaseModel):
    reason: str
    seed_chunk_id: str | None = None
    seed_relative_path: str | None = None
    path: list[EvidencePathStep]
    confidence: float | None = None
    stale: bool = False
```

Example evidence path:

```text
seed: config/settings.py:12-20
  -> code_reads_env(DATABASE_URL)
  -> env_var:DATABASE_URL
  -> config_defines_env
  -> docker-compose.yml:8-12
```

Diagnostics:

```python
class RepoContextDiagnostics(BaseModel):
    fusion_method: Literal["weighted_rrf"]
    rrf_k: int
    branch_diagnostics: dict[str, BranchDiagnostics]

    filter_diagnostics: FilterDiagnostics
    graph_diagnostics: GraphDiagnostics | None

    exact_anchor_analysis: ExactAnchorAnalysis
    scope_distribution: dict[str, int]
    file_distribution: list[FileDistributionEntry]

    matched_terms: list[str]
    uncovered_terms: list[str]

    stale_index: bool
    stale_graph: bool
    lifecycle_blocking: bool
```

Recommended actions should be small and operationally useful:

```text
read_file_range
run_local_rg
rebuild_index
build_graph
retry_with_graph_expand
retry_with_path_filter
retry_with_docs_scope
```

Do not include hidden reasoning. Do not include a final answer. Do not claim
semantic results are verified enough to edit.

## 8) Status and freshness architecture

Keep `index_status.v2` as the central status tool. Add a compact graph and
retrieval summary to it, and add a separate `graph_status()` for detailed graph
state.

### 8.1 Additive `index_status.v2` fields

Add:

```python
class RetrievalContractStatus(BaseModel):
    dense_available: bool
    sparse_available: bool
    sparse_contract_compatible: bool
    fusion_methods_supported: list[str]
    payload_filter_pushdown_supported: bool
    legacy_lexical_fallback_enabled: bool = False
    unavailable_codes: list[str] = []

class GraphStatusSummary(BaseModel):
    available: bool
    state: Literal[
        "missing",
        "ready",
        "partial",
        "stale",
        "incompatible",
        "building",
        "error",
    ]
    expansion_allowed: bool
    schema_version: int | None
    built_commit: str | None
    built_at: str | None
    extractor_versions_hash: str | None
    file_coverage_ratio: float | None
    stale_path_count: int | None
    invalidated_paths_preview: list[str] = []
    warning_codes: list[str] = []
```

`index_status.v2` should be enough for an agent to decide whether
`repo_context_search` can use dense, sparse, and graph branches. `graph_status()`
should return detailed node/edge counts, extractor coverage, stale file lists,
edge-type counts, and graph contract metadata.

### 8.2 Freshness rules

Use these rules:

```text
git freshness stale:
  search still allowed by default
  warning
  exact verification recommended

lifecycle stale:
  search blocked under current lifecycle contract

embedding contract mismatch:
  search blocked

sparse contract mismatch:
  dense search allowed
  hybrid/sparse branch unavailable
  explicit rebuild recommended

graph missing:
  dense+sparse search allowed
  graph branch unavailable

graph stale:
  dense+sparse search allowed
  graph branch skipped for stale seed paths
  graph warning in response

graph incompatible:
  graph branch unavailable
  rebuild_graph recommended
```

The P3 distinction between nonblocking git freshness stale, lifecycle-stale
blocking, and embedding-contract blocking carries forward unchanged.

## 9) Exact-search policy

Local `rg` remains the authority for exact identifiers, literals, env vars, route
names, SQL names, errors, and paths when the frontier coding agent has filesystem
access. MCP should not shell out to `rg` by default.

`repo_context_search` should still detect exact anchors and return query-time
guidance:

```text
verification.required = true
verification.reason = "exact_anchor_detected"
recommended_next_actions += run_local_rg(...)
```

Example:

```json
{
  "code": "run_local_rg",
  "severity": "warning",
  "detail": "Query contains exact env var DATABASE_URL. Verify indexed results against working tree before editing.",
  "argv_hint": ["rg", "--fixed-strings", "--line-number", "DATABASE_URL", "."]
}
```

Do not interpolate raw user text into a shell string. If hints are returned,
return argument arrays, not shell commands.

`indexed_exact_search` can be added after P4.C if MCP-only clients need it. It
must search only indexed chunk payloads, not the working tree, and must clearly
warn when the index is stale. It should not support regex in v1. It should use
identifier-aware word boundaries, smart-case behavior, bounded snippets, and
bounded matches per file.

## 10) Advanced stages

### 10.1 Reranker

Add reranker only after dense+sparse RRF and `repo_context_search` are stable.

Policy:

```text
rerank_mode="off" by default
rerank_mode="on" explicit
rerank_mode="auto" only after measured acceptance
```

Rerank over a bounded slate only:

```text
rerank_input_limit <= 50
```

Expose:

```text
reranker_available
reranker_model
reranker_contract
reranker_latency_ms
reranked_count
```

Do not silently turn rerank on for legacy tools.

### 10.2 ColBERT / multivectors

ColBERT-style late interaction is a P6+ feature. It changes index schema,
storage cost, query latency, model dependencies, and status compatibility.

Recommended status fields:

```text
colbert_available
colbert_index_present
colbert_contract_compatible
multivector_schema_version
```

Do not make ColBERT a requirement for `repo_context_search.v1`.

### 10.3 Tree-sitter

Tree-sitter is the right long-term extraction substrate, but it should not block
graph v1. First graph slice should reuse current chunkers, Python AST metadata
where available, Markdown headings, config parsers, and exact scans.

### 10.4 Typed PPR

Typed PPR can be useful after the graph has enough reliable edges and edge
weights. It should not be the v1 graph scorer. Start with bounded 1-hop
deterministic expansion and high-degree penalties.

HippoRAG's transferable idea is seeded propagation over a graph; literal
OpenIE/PPR-first indexing is not the right first architecture for code
repositories.

RepoHyper is closer to the intended code-side direction because it uses a
repo-level semantic graph with search/expand/refine retrieval for repository-
level code completion. That supports search-then-expand as an architectural
pattern, not a mandate to copy its full method.

## 11) Development vector

### Phase 0: Finish/lock P3 contracts

Goal: make current MCP stable enough to evolve.

Deliverables:

```text
index_status.v2 additive fields
search_v2 envelope
SearchFilters
query-centered snippets
matched_terms / why_matched / line_matches
max_results_per_file
exact fallback guidance
tests for empty-result compatibility
```

Acceptance:

```text
agents can determine readiness without prose parsing
legacy tools remain backward-compatible
no lifecycle operation runs implicitly
search_v2 carries response-level diagnostics
```

### Phase 1: P4.A sparse vectors + RRF

Goal: replace local BM25 cache as the production lexical branch.

Deliverables:

```text
Qdrant sparse vector schema
sparse manifest
lexical analyzer contract
sparse vector upsert during indexing
dense branch retrieval
sparse branch retrieval
service-level weighted RRF
branch diagnostics
status sparse contract
```

Acceptance:

```text
hybrid search no longer scrolls the collection for lexical recall
exact-like identifiers appear through sparse retrieval
dense query remains raw
sparse query is code-normalized
branch ranks and branch scores are visible
old indexes report sparse_missing instead of pretending full hybrid capability
```

Kill criteria:

```text
if sparse indexing requires an opaque neural model dependency for v1, defer it
if query-time sparse requires full corpus scroll, it is not P4.A-complete
```

### Phase 2: P4.B/P4.C lexical normalization + payload filters

Goal: make filtered retrieval predictable for agents.

Deliverables:

```text
code-aware lexical normalization
original-token preservation
exact anchor detection
payload indexes
Qdrant-side filters for simple fields
client-side glob filters with best-effort diagnostics
path_prefix boundary semantics
```

Acceptance:

```text
snake/camel/kebab/dotted/path tokens are searchable
env vars/routes/SQL names/file paths preserve exact forms
scope/language/chunk_type/domain/file_extension filters push down when possible
glob post-filtering reports under-recall risk
```

### Phase 3: P4.D `repo_context_search` without graph

Goal: introduce the future agent-facing envelope before graph complexity.

Deliverables:

```text
repo_context_search.v1
multi-query input
original-query injection
dense+sparse branch diagnostics
RRF fusion diagnostics
scope/file distribution
matched/uncovered terms
recommended next actions
flat results + file_groups summary
```

Acceptance:

```text
agent can decide which files to read next from one response
response is grounded by chunk_id/path/line/snippet
no hidden LLM planner
no graph dependency
legacy search tools unchanged
```

This phase is important. Do not wait for graph to introduce
`repo_context_search`; otherwise the first graph implementation will be
entangled with API, response-shape, and ranking changes.

### Phase 4: P5.A graph artifact + graph status

Goal: build graph state without changing retrieval quality yet.

Deliverables:

```text
SQLite graph store
graph metadata
graph file state
graph nodes/edges
graph_node_terms
graph_node_chunks
graph_status()
index_status.v2 graph summary
build_graph/rebuild_graph explicit tools
```

Initial extraction:

```text
file nodes
chunk nodes
symbol nodes from current metadata
doc_section nodes
config_item nodes
env_var nodes
policy_scope nodes
containment edges
doc/config/env/policy/reference edges
```

Acceptance:

```text
graph can be built explicitly
graph freshness is visible
per-file invalidation works
graph missing/stale does not block dense+sparse retrieval
graph is not used by search yet except diagnostics
```

### Phase 5: P5.B/P5.C seed binding + 1-hop expansion

Goal: make graph useful but bounded.

Deliverables:

```text
seed binding by chunk_id/path/symbol/matched terms
graph_mode off|auto|expand
1-hop allowlisted expansion
graph path scoring
graph branch ranking
RRF dense+sparse+graph fusion
evidence_paths
graph diagnostics
stale-edge suppression
```

Acceptance:

```text
env var query can find config/runtime/deploy context
docs/spec query can find referenced code context
policy query can find affected path context
graph result explains seed and path
simple exact symbol lookup is not polluted by graph noise
stale graph edges near changed files are skipped or warned
```

### Phase 6: P5.D better code edges

Goal: add high-value structural edges after the graph route is stable.

Deliverables:

```text
Python import edges
basic test-target edges
docs-to-code references
route edges for selected frameworks
config/env extraction improvements
migration/db edges only after explicit design
```

Acceptance:

```text
route -> handler -> tests works on supported frameworks
module -> imports -> tests improves context without high-degree noise
unsupported languages/frameworks degrade gracefully
```

### Phase 7: P6+ advanced retrieval

Goal: add expensive features only after baseline quality is measurable.

Candidates:

```text
cross-encoder rerank
ColBERT multivectors
tree-sitter multi-language extraction
typed PPR
SQL/migration/db object graph
trace-based calibration
```

Acceptance gate:

```text
feature improves repo-level context tasks
latency/cost visible in diagnostics
status compatibility explicit
off by default until measured
legacy tools unaffected
```

## 12) Resolved P5 open decisions

| Question | Decision |
| --- | --- |
| New graph route name | Use `repo_context_search`, not `search_v3`. |
| Graph status location | Add compact summary to `index_status.v2`; add detailed `graph_status()`. |
| `graph_mode=auto` triggers | Deterministic query/anchor/seed signals only; no LLM router. |
| Exact anchors and graph | Suppress graph for plain symbol/literal/path lookup; allow narrow graph for infra anchors like env vars, routes, SQL/config/deploy. |
| Graph lifecycle | Explicit `build_graph` / `rebuild_graph`; optional explicit inclusion in `build_index` only after stable. |
| Graph build timing | Separate in initial P5; profile-default subphase later. |
| Graph artifact split | One graph per repo/profile, not code/docs split. |
| Smallest useful edge set | file/chunk/symbol/doc/config/env/policy nodes with containment/reference/env/policy edges. |

## 13) Things not to build

Do not build these into the default architecture:

```text
hidden ReAct loop
hidden LLM query planner
final_answer tool
GraphRAG summary generator
OpenIE triple index as core code graph
Neo4j/graph database dependency in v1
query-time full collection BM25 scroll as production hybrid search
direct blending of dense/BM25/graph/reranker scores
automatic rebuild/reindex/runtime switch from status/search
shell-backed rg inside MCP by default
PRF/HyDE/LLM rewrite default path
ColBERT as baseline requirement
tree-sitter as graph-v1 blocker
typed PPR before deterministic 1-hop graph proves useful
```

These would make the MCP slower, less predictable, harder to operate, or less
useful to frontier coding agents.

## 14) Final acceptance criteria

The architecture is successfully realized when:

```text
1. Legacy tools remain backward-compatible.

2. index_status.v2 tells agents:
   runtime state,
   backend state,
   dense/sparse compatibility,
   graph availability,
   lifecycle state,
   freshness policy,
   explicit next actions.

3. hybrid retrieval uses Qdrant dense+sparse branches and RRF,
   not local query-time BM25 scroll and raw score blending.

4. dense input remains raw;
   sparse input uses code-aware lexical normalization
   while preserving exact anchors.

5. repo_context_search returns:
   grounded file/chunk/line results,
   origin branches,
   origin queries,
   matched/uncovered terms,
   evidence paths,
   route diagnostics,
   freshness warnings,
   recommended next actions.

6. graph expansion is optional, deterministic, bounded, and explainable.

7. stale graph state never silently influences results around edited files.

8. exact identifiers/literals still route agents toward local rg verification.

9. rerank, ColBERT, tree-sitter, and PPR are optional later stages with explicit
   status contracts and measurable value.
```

Shortest implementation path:

```text
P3 contracts
  -> Qdrant sparse vectors + RRF
  -> lexical normalization + payload filters
  -> repo_context_search dense+sparse envelope
  -> SQLite graph artifact + graph_status
  -> seed-bound 1-hop graph expansion + evidence paths
  -> richer edges
  -> optional rerank/ColBERT/tree-sitter/PPR
```

## 15) Source links

- [Qdrant Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)
- [RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion](https://arxiv.org/abs/2403.06095)
