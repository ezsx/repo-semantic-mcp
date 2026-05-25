# P4: RAG stack transfer from `rag_app` to code-repo search

## 1) Purpose

This note captures what we want to borrow from `a separate local `rag_app` project` for
`repo-semantic-search`.

The goal is not to build another research RAG project inside this MCP. The goal is
to turn the current "embed chunks and search vectors" implementation into a
practical retrieval layer that a frontier coding agent can use every day in large
repositories.

The frontier model remains the planner and final reasoning engine. The MCP should
provide fast, grounded, inspectable retrieval over code, docs, tests, config, and
deploy files.

## 1.1) Research boundary on `rag_app`

For architecture work, `rag_app` has been studied enough. We do not need another
long pass over that repository before moving to HippoRAG research.

What was read and used:

- project README and agent policy, to understand the intended RAG route and
  operational constraints;
- architecture/progress/spec notes, including the ablation summary that explains
  why the final stack settled on dense+sparse recall, RRF, ColBERT, rerank, and
  original-query injection;
- `hybrid_retriever`, Qdrant store, search tool, rerank service, GPU server,
  query planner, query signals, compose context, coverage, refinement, tool
  runner, and agent visibility code;
- schema files around search plans, candidates, and metadata filters.

The parts that matter for `repo-semantic-search` are now clear:

- retrieval route: dense raw query plus sparse lexical branch, fused by RRF;
- lexical branch: normalize only lexical/sparse input, never dense input;
- candidate strategy: wide lexical recall, dense recall, then fusion/rerank;
- grounding: stable ids, snippets, line references, citations/coverage-style
  diagnostics;
- operational lesson: optional advanced stages must be explicit in status and
  schema contracts.

What does not need more study right now:

- the full ReAct application loop;
- platform-specific message metadata and analytics tools;
- long evaluation harness details;
- NLI/final-answer support checks;
- exact model-serving implementation details beyond knowing the available
  endpoints and model roles.

We may return to `rag_app` later only when implementing a specific borrowed
component, for example sparse vector indexing, RRF merge, ColBERT multivectors,
or reranker backend wiring.

## 2) What the current MCP already has

The current implementation already has the right operational base:

- Qdrant-backed persistent collections split into `code` and `docs`.
- Repository-aware chunking with line ranges, `chunk_type`, `symbol_path`,
  `heading_path`, and path-derived `domain_tags`.
- Dense embedding search through the configured embedding backend.
- Hybrid search contract with lexical explanations, line matches, snippets, path
  filters, and `search_v2`.
- Agent-grade `index_status.v2` diagnostics, freshness, backend/runtime checks,
  and lifecycle boundaries.

The main retrieval weakness is that hybrid search is still local and shallow:

- lexical recall is computed by scrolling chunks into an in-process BM25 cache;
- dense and lexical scores are blended as `dense * 0.7 + lexical * 0.3`;
- Qdrant does not store a sparse vector branch yet;
- the API can return good rows, but it does not yet provide a stronger retrieval
  route with multi-query merge, route diagnostics, or coverage/gap signals.

## 3) What matters from `rag_app`

The useful route in `rag_app` is:

```text
query
  -> optional query_plan / subqueries
  -> dense recall + BM25 sparse recall
  -> weighted RRF
  -> optional ColBERT late interaction
  -> optional cross-encoder relevance filter / re-sort
  -> MMR-style merge / dedupe
  -> grounded context envelope with citations and coverage diagnostics
```

For this MCP we should port the retrieval stack, not the whole agent.

### 3.1 Dense plus sparse recall

`rag_app` uses two independent recall branches:

- dense query stays raw;
- sparse query is normalized through a lexical normalization path;
- sparse recall is wider than dense recall;
- both branches are executed inside Qdrant through named dense/sparse vectors.

For code search this is more important than in ordinary text search. Agents often
ask for a concept, but the useful anchor can be an exact symbol, env var, route
path, CLI flag, migration name, error string, table name, or file path fragment.

Transfer target:

- add a Qdrant sparse vector branch for indexed chunks;
- stop rebuilding BM25 from a full collection scroll during query execution;
- keep dense text raw;
- apply code-aware normalization only to the lexical/sparse branch.

### 3.2 Rank fusion instead of score blending

`rag_app` uses RRF over dense and sparse candidate lists. Its measurements showed
that the exact RRF weights were less important once late interaction existed, but
the structure itself was robust.

The current MCP blends unrelated score scales. That is fragile because Qdrant
cosine scores, BM25 scores, normalized local BM25 scores, and future reranker
scores are not naturally comparable.

Transfer target:

- rank dense candidates and sparse candidates separately;
- fuse by RRF, initially with sparse-biased weights similar to `rag_app`;
- preserve per-branch scores in the response for diagnostics;
- expose `fusion_method`, candidate counts, and branch availability in
  `SearchDiagnostics`.

### 3.3 Late interaction and rerank as optional layers

`rag_app` gets a large quality gain from ColBERT and uses a cross-encoder as a
CRAG-style relevance filter/re-sort.

For code-repo MCP, this should not be the first implementation dependency. It
changes index schema, vector storage, runtime latency, model setup, and hardware
expectations.

Transfer target:

- design the route so `colbert` and `reranker` can be plugged in later;
- do not require them for the first stronger retrieval slice;
- record schema/runtime compatibility in status before any multivector rollout;
- if rerank is added, treat it as an explicit optional stage over a bounded
  candidate set, not as a hidden behavior change in legacy tools.

### 3.4 Multi-query merge, but not LLM planning inside MCP

`rag_app` benefits from query planning only when the original query is injected
back into the search set. Without the original query, lexical anchors are lost.

For this project, the frontier coding agent can produce search plans itself. MCP
does not need a slow LLM planner.

Transfer target:

- support `queries: list[str]` or `subqueries: list[str]` in a future
  `search_v3` / `repo_context_search`;
- always include the original query as the first retrieval query;
- merge results with RRF/MMR-style diversity;
- expose which query found each result;
- optionally provide deterministic route templates, but not LLM planning.

### 3.5 Grounding envelope for agents

`rag_app` makes search useful to an agent by carrying citations, compact context,
coverage, and structured tool diagnostics.

For code agents, the equivalent is not a final answer tool. It is a retrieval
envelope that makes the next file-read or exact verification obvious.

Transfer target:

- keep result rows grounded by `chunk_id`, `relative_path`, `start_line`,
  `end_line`, `snippet_start_line`, and `snippet_end_line`;
- add route-level diagnostics such as matched query terms, branch coverage,
  scope distribution, file diversity, score gap, and uncovered terms;
- make `read_chunk` / direct file reads the expected next step for edits;
- keep exact search as a required verification path for identifiers and literals.

## 4) What not to port

Do not port these parts directly:

- Full ReAct loop. The MCP is a tool for frontier agents, not the agent runtime.
- LLM query planner as a default retrieval dependency.
- `final_answer`, NLI support checks, or answer-generation tools.
- platform-specific message metadata such as channels, authors, dates, and analytics.
- Long ablation workflow as a project dependency.
- PRF, HyDE, or LLM rewrite as default behavior. These can erase code anchors.

The guiding rule: if the frontier model can do it cheaply and flexibly, the MCP
should expose primitives and diagnostics instead of hiding another agent inside.

## 5) Code-search adaptation

A code repository has different retrieval units than a message corpus.

Important indexed dimensions:

- file: `relative_path`, extension, generated flag, scope;
- symbol: function, method, class, module preamble, structured config key;
- code role: runtime, docs, tests, config, deploy, migrations, scripts;
- topology hints: imports, call sites, test targets, config/env ownership;
- agent policy: `AGENTS.md`, `agent_context/**`, repo-owned guidance files.

Common query intents:

- "where is this endpoint implemented";
- "find docs/spec for this flow";
- "find tests around this module";
- "find config/env/deploy contract";
- "find migration/schema owner";
- "find exact symbol/literal/error/route";
- "find cross-file flow across API, service, docs, and tests".

The route should support both conceptual discovery and exact anchors:

```text
agent query or agent-provided subqueries
  -> dense raw query
  -> lexical code-normalized query
  -> Qdrant dense prefetch + sparse prefetch
  -> weighted RRF
  -> optional diversity by file/scope/role
  -> optional rerank/late interaction
  -> SearchResponse envelope
  -> read_chunk / file read / exact verification
```

## 6) Proposed implementation slices

### Slice P4.A: Qdrant sparse branch and RRF

Add sparse lexical vectors to the Qdrant index and query them as a first-class
retrieval branch.

Expected changes:

- collection schema adds named dense and sparse vector configs;
- index upsert writes both dense and sparse vectors;
- `hybrid_search` no longer needs to scroll all chunks into local BM25;
- fusion changes from score blend to RRF;
- diagnostics include dense/sparse candidate counts and fusion method;
- status/index contract reports whether sparse vectors are present.

This is the main transfer from `rag_app`.

### Slice P4.B: Code-aware lexical normalization

Build a lexical normalization path for code search.

Expected behavior:

- split snake_case, kebab-case, dotted paths, slashes, route paths, and camelCase;
- preserve exact tokens for symbols, env vars, route names, table names, and file
  paths;
- normalize only the lexical branch;
- keep dense input raw.

This mirrors the `rag_app` sparse-only normalization lesson.

### Slice P4.C: Indexed payload filters

Move important filters closer to Qdrant.

Expected changes:

- create payload indexes for `relative_path`, `scope`, `language`, `chunk_type`,
  `domain_tags`, `is_generated`, and possibly `file_extension`;
- translate supported search filters into Qdrant filters before retrieval;
- keep client-side glob filtering for patterns Qdrant cannot represent cleanly;
- report best-effort filtering when post-filtering can under-recall.

### Slice P4.D: Agent-facing context search

Add a higher-level search contract while keeping existing tools compatible.

Candidate tool name:

- `repo_context_search`
- or `search_v3`

Expected parameters:

- `query: str`
- `subqueries: list[str] = []`
- `route: "auto" | "semantic" | "hybrid" | "exact_handoff"`
- current filters from `search_v2`
- optional `result_diversity` controls

Expected response additions:

- `route_used`
- `queries_used`
- `branch_diagnostics`
- `scope_distribution`
- `file_distribution`
- `matched_terms`
- `uncovered_terms`
- `recommended_next_steps`

This gives agents the useful parts of `rag_app` compose/coverage without taking
over final reasoning.

### Slice P4.E: Optional rerank and late interaction

Only after the sparse/RRF baseline is stable:

- add an optional reranker backend role;
- add an optional ColBERT backend role and multivector schema;
- expose explicit runtime/status compatibility;
- keep the feature off unless configured;
- measure with lightweight repo-specific acceptance tasks before making it
  recommended.

## 7) HippoRAG fit

HippoRAG should be evaluated after the P4 transfer baseline is clear.

It should not replace dense+sparse retrieval. It should become an optional graph
expansion layer:

```text
dense+sparse seeds
  -> graph expansion over files, symbols, imports, tests, docs, config, deploy
  -> rerank/fusion with original candidates
  -> grounded SearchResponse
```

Likely graph nodes:

- file;
- chunk;
- symbol;
- import/module edge;
- call/reference edge;
- test-to-target edge;
- docs/spec-to-runtime edge;
- config/env-to-runtime edge;
- migration/schema-to-runtime edge.

The core question for HippoRAG research is therefore narrow:

Can graph expansion improve agent context gathering for cross-file development
tasks without making simple symbol/config searches slower or noisier?

## 8) Success criteria

The first useful milestone is not a benchmark suite. It is a better daily tool.

Minimum acceptance for P4.A-P4.D:

- exact-like identifiers appear through the sparse branch without scanning the
  whole collection in Python;
- conceptual queries still return semantically relevant code/docs;
- path and role filters remain predictable;
- every result explains which branch/query found it;
- agents can decide the next read/rg action from the response alone;
- stale/runtime/lifecycle safety from P3 remains unchanged;
- old search tools keep their existing public behavior.
