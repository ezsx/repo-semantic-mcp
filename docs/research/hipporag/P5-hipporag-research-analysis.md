# P5: HippoRAG research report analysis

Source report:
`<downloaded-deep-research-report.md>`

## 1) Verdict

The report is strong enough to become architecture input.

Its most important conclusion is correct: we should not build literal
"HippoRAG for code" with OpenIE triples as the core index. We should build
`repo-semantic-search` as dense+sparse repository retrieval with HippoRAG-style
seeded graph expansion.

The future shape is:

```text
query / optional subqueries
  -> dense raw retrieval
  -> sparse code-normalized retrieval
  -> RRF seed fusion
  -> optional typed graph expansion from strong seeds
  -> RRF/fusion of dense+sparse+graph branches
  -> optional rerank / optional ColBERT later
  -> grounded response with file lines and structural reasons
```

This keeps the MCP as a retrieval layer for frontier coding agents. It does not
turn it into an internal graph agent, answer generator, or hidden query planner.

## 2) Checked claims

The report's main claims line up with primary sources and implementation facts:

- HippoRAG v1 combines LLM-extracted knowledge graphs with Personalized
  PageRank for multi-hop retrieval. Its transferable idea is seeded graph
  propagation, not OpenIE as a mandatory implementation detail.
  Source: <https://huggingface.co/papers/2405.14831>

- HippoRAG 2 adds passage/context nodes, uses passage and triple seeds, applies
  recognition memory over triples, and keeps graph retrieval as a retrieval aid.
  This supports our need for chunk/file context nodes, not symbol-only graphs.
  Source: <https://openreview.net/pdf?id=LWH8yn4HS2>

- RepoHyper supports the search-then-expand direction for code: it builds a
  repo-level semantic graph, searches, expands, and refines/reranks retrieved
  context. This is closer to our desired route than GraphRAG-style summaries.
  Source: <https://arxiv.org/abs/2403.06095>

- RepoGraph confirms that repository-level graph context can improve software
  engineering workflows, but it is not proof that we should expose a raw graph
  database or hidden graph-query agent.
  Source: <https://arxiv.org/abs/2410.14684>

- Qdrant already supports the dense+sparse+RRF and multistage retrieval direction
  we need. It also supports ColBERT-style multivectors for later reranking.
  Sources:
  <https://qdrant.tech/documentation/search/hybrid-queries/>
  <https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/>

## 3) Decisions accepted from the report

### 3.1 Graph is a branch, not the root

The graph should not replace dense+sparse retrieval. It should start from strong
seeds returned by the dense+sparse RRF baseline.

This gives us:

- stable fallback when graph is missing or stale;
- less latency for simple exact/symbol queries;
- better explainability because every graph result can point back to a seed;
- compatibility with existing `search_v2` behavior.

### 3.2 `graph_mode` should be explicit

Use a future parameter:

```text
graph_mode = "off" | "auto" | "expand"
```

Suggested behavior:

- `off`: dense+sparse only.
- `auto`: deterministic graph expansion only when triggers say it is useful.
- `expand`: agent explicitly asks for related structural context.

This is better than an internal LLM router. Frontier models can plan; MCP should
keep routing deterministic and visible.

### 3.3 Code graph should not be OpenIE

For code repositories, graph edges should come from code structure and repo
metadata:

- containment;
- definitions;
- imports;
- tests;
- docs/spec references;
- config/env usage;
- deploy files;
- migrations/schema;
- repo-owned agent policy.

OpenIE triples may be useful for prose docs later, but they should not be a v1
dependency.

### 3.4 Results remain chunk-grounded

The primary result object should stay grounded in files/chunks:

- `relative_path`;
- `start_line`;
- `end_line`;
- snippet range;
- `chunk_id`;
- symbol/heading metadata;
- evidence paths.

Graph nodes explain why a result is relevant. They should not replace grounded
file/line results.

### 3.5 Graph freshness is its own contract

Graph state must be visible in status:

- graph available/missing/stale/partial;
- graph schema version;
- built commit;
- built timestamp;
- extractor versions;
- file hash coverage;
- invalidated paths preview.

If the graph is stale, the dense+sparse route can still work, but graph expansion
must be skipped or marked best-effort. The MCP must not silently use stale graph
edges around edited files.

## 4) Scope corrections

The report's direction is right, but the first implementation slice should be
smaller than its Phase 1 suggestion.

### 4.1 Do not combine sparse vectors and graph in one implementation slice

Server-side sparse vectors + RRF are the retrieval baseline. They should land
before graph expansion.

Recommended split:

- P4: dense+sparse Qdrant retrieval, RRF, lexical normalization, payload filters.
- P5: graph artifact, graph status, seed binding, graph expansion.

This keeps risk bounded. Otherwise we would debug sparse vector schema, graph
schema, ranking, and response shape at the same time.

### 4.2 Phase 1 graph should be smaller

The report suggests v1 graph types:

```text
file/chunk/symbol/import/test/doc/config/env/policy
```

That is directionally right but too broad for the first graph slice.

Better first graph slice:

- `file`;
- `chunk`;
- `symbol` for Python only or current structured chunkers;
- `doc_section`;
- `config_item`;
- `env_var`;
- `policy_scope`;
- containment/reference edges built from already indexed payloads and exact scans.

Then add imports/test-target extraction in the next slice once the graph artifact
and response path are stable.

### 4.3 Tree-sitter should be planned, not forced immediately

Tree-sitter is the right long-term multi-language parser layer, but it adds
packaging and grammar management. The first useful graph can reuse current
chunkers, Python AST metadata, Markdown headings, structured config parsers, and
exact scans.

Tree-sitter should be a P5/P6 expansion, not a blocker for graph status and
1-hop evidence paths.

### 4.4 Exact-anchor gating needs nuance

The report says strong exact anchors can disable graph expansion. That is mostly
right for literal lookup, but some exact anchors are exactly where graph helps:

- env var -> config loader -> compose/deploy docs;
- route literal -> handler -> tests/spec;
- SQL table -> migration -> repository;
- error string -> raising code -> tests/docs.

So the rule should be:

- exact symbol/literal lookup defaults to graph off;
- exact infrastructure anchors may trigger a narrow allowlisted graph expansion;
- `graph_mode=expand` always lets the agent request related context explicitly.

### 4.5 SQLite is probably enough for v1

The report mentions SQLite or DuckDB. For v1 adjacency queries, SQLite is the
better default:

- available everywhere;
- simple transactions;
- good enough for node/edge tables and indexes;
- easy to invalidate per file.

DuckDB can wait until we need analytics-style graph summaries.

## 5) Proposed architecture after the report

### 5.1 Storage layers

Keep two storage roles:

- Qdrant: dense vectors, sparse vectors, optional ColBERT multivectors, payload
  filters, chunk payloads.
- SQLite graph artifact: repo graph nodes, edges, node-to-chunk mapping,
  per-file graph state, extractor metadata.

Do not introduce a graph database in the first architecture.

### 5.2 Graph tables

Initial tables:

```text
graph_metadata(
  repo_root,
  schema_version,
  built_commit,
  built_at,
  extractor_versions_json
)

graph_files(
  relative_path,
  content_hash,
  mtime,
  graph_state,
  indexed_at
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

Useful indexes:

- `node_type, key`;
- `relative_path`;
- `chunk_id`;
- `source_node_id, edge_type`;
- `target_node_id, edge_type`;
- `relative_path, content_hash`.

### 5.3 Initial node types

Start with:

- `file`;
- `chunk`;
- `symbol`;
- `doc_section`;
- `config_item`;
- `env_var`;
- `policy_scope`.

Later:

- `route`;
- `test`;
- `fixture`;
- `db_object`;
- `migration`;
- `package`;
- `import_target`.

### 5.4 Initial edge types

Start with:

- `file_contains_chunk`;
- `chunk_defines_symbol`;
- `file_contains_doc_section`;
- `doc_references_symbol`;
- `doc_references_config`;
- `config_defines_env`;
- `code_reads_env`;
- `policy_applies_to_path`;
- `chunk_mentions_path_or_symbol`.

Later:

- `file_imports_module`;
- `test_targets_symbol`;
- `route_handled_by_symbol`;
- `migration_touches_db_object`;
- `symbol_uses_db_object`;
- `symbol_calls_symbol`.

### 5.5 Query route

Future route:

```text
repo_context_search(query, subqueries=[], graph_mode="auto", filters=...)
  1. Run dense raw branch.
  2. Run sparse code-normalized branch.
  3. RRF dense+sparse into seed candidates.
  4. Bind top seeds to graph nodes through chunk_id/path/symbol/matched terms.
  5. If graph is available and graph_mode allows:
       expand 1 hop over allowlisted edge types;
       score graph candidates with path scoring;
       convert graph candidates back to chunks/files;
       rank graph branch.
  6. RRF dense+sparse+graph branches.
  7. Optional rerank over a small slate.
  8. Return grounded results with compact evidence paths.
```

### 5.6 Graph scoring v1

Start with deterministic path scoring:

```text
path_score =
    seed_rank_prior
  + exact_anchor_bonus
  + edge_type_weight
  + multi_seed_support_bonus
  - hop_penalty
  - high_degree_penalty
  - generated_or_vendor_penalty
```

Then transform graph candidates into ranks and use rank fusion. Do not blend
raw graph scores with dense/sparse scores.

Typed PPR is a later scorer, not the first graph implementation.

## 6) Suggested implementation sequence

### P4.A: Qdrant sparse vectors and RRF

- Add named dense/sparse vectors.
- Upsert sparse vectors during indexing.
- Replace local BM25 cache with Qdrant sparse retrieval.
- Fuse dense/sparse by RRF.
- Add diagnostics for branch counts and fusion method.

### P4.B: Code-aware lexical normalization

- Normalize only sparse input.
- Split code identifiers and paths.
- Preserve original query and exact anchors.

### P4.C: Qdrant payload filters

- Add payload indexes.
- Push compatible filters into Qdrant.
- Keep glob filters client-side when needed.

### P5.A: Graph artifact and graph status

- Add SQLite graph store.
- Add graph schema metadata.
- Build file/chunk/symbol/doc/config/env/policy nodes from existing index data.
- Add graph freshness fields to status.

### P5.B: Seed binding and 1-hop expansion

- Bind search result chunks to graph nodes.
- Expand over allowlisted 1-hop edges.
- Return graph candidates as ranked chunks.
- Add `graph_mode`.

### P5.C: Evidence paths in response

- Add compact `evidence_paths`.
- Add `origin_branches`.
- Add `graph_diagnostics`.
- Add `recommended_next_actions`.

### P5.D: Better code edges

- Add import edges.
- Add test-target edges.
- Add docs-to-code references.
- Add config/env extraction improvements.

### P6+: Advanced graph retrieval

- Tree-sitter multi-language extraction.
- Route/framework extractors.
- SQL/migration/db object edges.
- Typed PPR scorer.
- Optional ColBERT/rerank after graph fusion.
- Trace-based calibration.

## 7) Open decisions before final architecture

These should be resolved in the final architecture pass:

1. Should the new graph route be a new tool (`repo_context_search`) or
   `search_v3`?
2. Should graph status be part of `index_status.v2` as additive fields or a new
   `graph_status()` tool?
3. What exact triggers enable `graph_mode=auto`?
4. Which exact anchors should suppress graph expansion, and which should trigger
   narrow infrastructure graph expansion?
5. How should graph lifecycle operations map to existing lifecycle policy?
6. Should graph build happen during index build, or as a separate explicit
   lifecycle operation?
7. Should graph be one artifact per repo/profile, or split by code/docs scope?
8. What is the smallest useful edge set for the first implementation?

## 8) Final reading

The report changes the next architecture in a useful way:

- P4 remains dense+sparse retrieval modernization.
- P5 becomes graph artifact and seeded graph expansion.
- HippoRAG contributes the retrieval pattern, not the literal OpenIE graph.
- RepoHyper is the strongest code-side confirmation of search-then-expand.
- Qdrant remains the vector retrieval substrate.
- SQLite graph artifact is enough for v1.
- No hidden graph-query LLM planner should be added to MCP.

This gives us a clean path to the final architecture prompt: "dense+sparse RRF
repo search with optional HippoRAG-style seeded typed graph expansion."
