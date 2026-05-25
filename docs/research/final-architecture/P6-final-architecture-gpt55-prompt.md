# P6 Prompt: Final Architecture for `repo-semantic-search`

Use this prompt with GPT-5.5 Pro after uploading the artifacts listed in
`docs/research/final-architecture/P6-final-architecture-artifact-index.md`.

```text
# Final Architecture Task: `repo-semantic-search` as dense+sparse+graph retrieval MCP

## Role

You are GPT-5.5 Pro acting as a principal RAG architect and senior software architect. Your task is to design the future architecture of `repo-semantic-search`, an MCP server for frontier coding agents working in large repositories.

This is not an implementation task. Produce a final architecture and development vector that we can later convert into precise implementation specifications. Be opinionated, concrete, and critical. If a proposed idea is too expensive, too vague, or likely to make the MCP worse for coding agents, say so directly.

---

## What this project is

`repo-semantic-search` is a reusable MCP server for repository search. Its users are frontier coding agents, not end-user chat users. The tool should let a model quickly gather the right context before editing a large codebase.

The product goal:

- help frontier models work in large repositories safely and efficiently;
- reduce time spent opening irrelevant files;
- find relevant code, docs, tests, configs, deployment contracts, migrations, schemas, symbols, routes, env vars, and cross-file flows;
- return grounded evidence with file paths, line ranges, snippets, matched terms, diagnostics, and freshness warnings;
- keep exact-search workflows available for identifiers, literals, routes, env vars, SQL names, and error strings;
- make index/runtime/lifecycle state understandable to agents;
- avoid implicit lifecycle operations like rebuild/reindex/runtime switch unless explicitly requested.

The MCP must remain a retrieval layer. The frontier model remains the planner, editor, and final reasoning engine. Do not design a hidden autonomous agent, hidden ReAct loop, hidden final-answer tool, or slow internal LLM query planner as the default path.

---

## Artifacts you should read

Required:

1. `docs/specifications/P3-agent-grade-status-search-and-exact-fallback-specification.md`
   - Current agent-grade status/search/exact fallback contract.
   - Defines `index_status.v2`, `search_v2`, filters, snippets, diagnostics, and lifecycle safety.

2. `docs/research/rag-transfer/P4-rag-stack-transfer-from-rag-app.md`
   - What we transfer from an existing successful RAG project.
   - Dense+sparse retrieval, lexical-only normalization, RRF, optional ColBERT/rerank, grounded response envelope.

3. `deep-research-report (8).md`
   - External deep research on HippoRAG / graph retrieval for code repository search.
   - Covers HippoRAG, HippoRAG 2, GraphRAG, LightRAG, RAPTOR, RepoHyper, RepoGraph, CodexGraph.

4. `docs/research/hipporag/P5-hipporag-research-analysis.md`
   - Our analysis of the HippoRAG report.
   - Converts research into accepted project decisions, scope corrections, storage options, graph tables, route, scoring, and open decisions.

Optional if context allows:

- `README.md`
- `docs/specifications/P1.Z-repo-semantic-search-mcp-specification.md`
- `docs/specifications/P2-repo-semantic-search-lifecycle-and-backend-abstraction-specification.md`
- current implementation files: `models.py`, `search_service.py`, `qdrant_store.py`, `indexer.py`, `chunkers/factory.py`, `mcp_server.py`.

---

## Current architectural baseline

The MCP already has:

- Qdrant-backed persistent vector collections;
- separate logical `code` and `docs` scopes;
- chunking for Python, Markdown, JSON/YAML/TOML, generic text;
- chunk metadata: `chunk_id`, `relative_path`, `scope`, `language`, `chunk_type`, `start_line`, `end_line`, `symbol_path`, `heading_path`, `domain_tags`, `is_generated`;
- `index_status.v2` with repo identity, active repo, branch/commit freshness, dirty/untracked info, backend/runtime checks, lifecycle state, compatibility contracts, and host-side hints;
- `search_v2` with include/exclude path filters, extension/language/chunk_type/domain filters, snippets, matched terms, line matches, warnings, diagnostics;
- semantic and hybrid search tools;
- explicit lifecycle safety: no automatic rebuild/reindex/runtime switch hidden behind status/search.

Current weaknesses:

- semantic retrieval is mostly dense nearest-neighbor chunk search;
- hybrid search still uses local BM25 over scrolled chunks instead of Qdrant sparse vectors;
- dense/lexical scores are blended directly;
- no real graph layer connects chunks/files/symbols/tests/docs/config/deploy/migrations;
- agents still have to infer connected context manually after finding a seed file;
- there is no final `repo_context_search` / `search_v3` architecture yet.

---

## What we already decided from the separate RAG project

We have a separate recent RAG project that already achieved strong results using:

- dense embeddings;
- sparse/BM25 recall;
- Qdrant;
- weighted RRF;
- ColBERT late interaction;
- cross-encoder rerank/filter;
- sparse-only lexical normalization;
- original-query injection;
- multi-query merge;
- grounded context envelope;
- coverage/refinement diagnostics;
- payload indexes and metadata filters.

We do not want to port the full app. We want to port the retrieval stack.

Accepted lessons:

1. Dense-only search is not enough.
2. Lexical/sparse recall is critical for code anchors.
3. Dense query text should remain raw.
4. Lexical/sparse query text should use code-aware normalization.
5. Original query must always be preserved.
6. RRF is safer than blending unrelated dense/BM25/reranker scores.
7. ColBERT/rerank are useful later, but should be optional explicit stages.
8. Grounding and diagnostics are part of the product, not nice-to-have details.
9. MCP should not become another hidden agent.

The expected non-graph baseline:

```text
query / optional subqueries
  -> dense branch with raw query
  -> sparse branch with code-aware lexical normalization
  -> Qdrant dense recall + Qdrant sparse recall
  -> weighted RRF
  -> optional rerank / optional ColBERT later
  -> grounded SearchResponse
```

Code-aware lexical normalization should preserve and split:

- snake_case;
- camelCase;
- kebab-case;
- dotted paths;
- slash paths;
- route paths;
- env vars;
- symbols;
- filenames;
- SQL table names;
- migrations;
- CLI flags;
- error strings;
- config keys.

---

## What the HippoRAG research suggests

The research report argues that HippoRAG is relevant, but not as a literal OpenIE graph implementation.

Accepted direction:

- do not replace dense+sparse retrieval with graph retrieval;
- use graph as a retrieval branch after strong seeds;
- keep graph expansion budgeted, typed, explainable, and freshness-aware;
- keep result rows grounded in chunks/files/lines;
- do not dump raw subgraphs into MCP responses;
- avoid hidden graph-query LLM planners;
- avoid summary-heavy GraphRAG/RAPTOR-style indexing as the main code-search route.

Recommended graph route:

```text
dense+sparse+RRF seeds
  -> bind seeds to repo graph nodes
  -> typed 1-hop or controlled 2-hop graph expansion
  -> graph branch ranking
  -> RRF/fusion with dense+sparse candidates
  -> optional rerank
  -> grounded response with evidence paths
```

Key idea:

Code repositories are highly connected. Useful context is often not semantically similar but structurally related:

- endpoint -> handler -> service -> repository -> schema/migration;
- function -> callers/callees;
- class -> methods -> tests;
- route -> docs/spec -> request/response schema -> integration test;
- env var -> config loader -> compose/deploy docs;
- SQL table -> model/schema -> migration -> repository methods;
- agent policy -> module-specific context -> implementation files.

But graph expansion can easily become noisy, stale, expensive, or over-broad. The architecture must define strict budgets, edge types, freshness rules, and response diagnostics.

---

## Decisions already leaning but not final

Please validate or correct these.

### Storage

Likely:

- Qdrant remains vector retrieval substrate for dense/sparse/optional ColBERT.
- SQLite graph artifact stores graph nodes/edges/freshness/extractor metadata.
- No graph database in v1.

### Graph mode

Likely future parameter:

```text
graph_mode = "off" | "auto" | "expand"
```

- `off`: dense+sparse only.
- `auto`: deterministic graph expansion if triggers say it helps.
- `expand`: explicit related structural context request.

### First graph scope

Possible v1:

- nodes: `file`, `chunk`, `symbol`, `doc_section`, `config_item`, `env_var`, `policy_scope`;
- edges: `file_contains_chunk`, `chunk_defines_symbol`, `file_contains_doc_section`, `doc_references_symbol`, `doc_references_config`, `config_defines_env`, `code_reads_env`, `policy_applies_to_path`, `chunk_mentions_path_or_symbol`;
- later: imports, tests, routes, migrations, db objects, call graph, tree-sitter, LSP/LSIF.

### Scoring

Likely:

- graph branch has its own deterministic path scoring first;
- convert graph scores to ranks;
- final fusion uses rank fusion, not raw score blending;
- typed PPR is a later optional scorer, not v1 default.

### Lifecycle/freshness

Graph status must be explicit:

- missing/fresh/partial/stale;
- graph schema version;
- built commit;
- built timestamp;
- extractor versions;
- file hash coverage;
- invalidated path previews;
- host action hints.

If graph is stale, search must degrade to dense+sparse or mark graph results best-effort. Do not silently use stale graph edges for edited files.

---

## Your task

Create the final architecture for future `repo-semantic-search`.

This architecture should answer:

1. What is the north-star architecture of the MCP?
2. What are the main subsystems?
3. What data stores exist and what belongs in each?
4. What is the indexing/build lifecycle?
5. What is the query lifecycle?
6. What are the MCP tools/contracts?
7. Should we add `repo_context_search`, `search_v3`, or extend `search_v2`?
8. What status/freshness contracts are needed?
9. What graph schema should v1 use?
10. What scoring/fusion/rerank route should be used?
11. What exact-search fallback/handoff remains?
12. What should not be built?
13. What implementation phases should follow?
14. What open questions remain before writing specs?

---

## Required output format

Return a structured architecture document with these sections.

### 1. Executive architecture verdict

State the final architectural direction in 5-10 bullets. Include what you reject.

### 2. North-star architecture

Describe the final desired system as a coherent whole.

Include a route diagram like:

```text
agent query
  -> status/freshness preflight
  -> dense raw branch
  -> sparse normalized branch
  -> RRF seed fusion
  -> graph expansion branch
  -> final fusion/rerank
  -> grounded response
  -> exact read/search verification
```

Correct this route if you think it is wrong.

### 3. Subsystems and responsibilities

Define each subsystem:

- settings/config;
- repo registry/lifecycle;
- indexer/chunkers;
- Qdrant vector store;
- sparse lexical encoder;
- lexical normalization;
- graph builder;
- graph store;
- graph freshness;
- search service;
- reranker/ColBERT optional services;
- MCP server tools;
- diagnostics/status.

### 4. Storage architecture

Define Qdrant schema and graph artifact schema.

For Qdrant:

- dense vectors;
- sparse vectors;
- optional ColBERT multivectors;
- payload fields;
- payload indexes;
- compatibility/versioning.

For graph:

- store choice;
- node tables;
- edge tables;
- metadata tables;
- per-file freshness;
- indexes;
- schema versioning;
- invalidation strategy.

### 5. Indexing and lifecycle architecture

Explain:

- full build;
- partial reindex;
- graph build;
- graph incremental update;
- watcher integration;
- dirty/untracked worktree behavior;
- branch/commit awareness;
- lifecycle commands vs status hints;
- compatibility with existing "no implicit lifecycle operations" policy.

### 6. Query architecture

Define:

- semantic route;
- hybrid dense+sparse route;
- graph route;
- exact-handoff route;
- multi-query behavior;
- original-query injection;
- filter behavior;
- branch diagnostics;
- fallback behavior when graph/sparse/rerank unavailable.

### 7. Graph retrieval design

Define:

- graph mode semantics;
- auto triggers;
- seed selection;
- seed-to-node binding;
- edge allowlists;
- hop budgets;
- high-degree penalties;
- generated/vendor filtering;
- graph candidate ranking;
- fusion into final results;
- evidence path compression.

Be specific enough that we can write a spec from it.

### 8. MCP tool/API contracts

Recommend whether to create:

- `repo_context_search`;
- `search_v3`;
- additive extension to `search_v2`;
- `graph_status`;
- `read_chunk`;
- any exact-search helper.

For each tool, define:

- purpose;
- parameters;
- response shape;
- compatibility story;
- when agents should use it.

### 9. Response model

Design the future response envelope:

- route used;
- queries used;
- results;
- origin branches;
- dense/sparse/graph ranks;
- matched terms;
- evidence paths;
- graph diagnostics;
- warnings;
- recommended next actions;
- exact-search/read handoff.

Keep it compact enough for MCP use.

### 10. Scoring and ranking

Define:

- RRF usage;
- branch weights;
- sparse candidate width vs dense candidate width;
- graph candidate scoring;
- rerank placement;
- ColBERT placement;
- when not to rerank;
- why raw score blending should or should not be avoided.

### 11. Exact search and verification story

Define how exact search fits with semantic/hybrid/graph search.

We already believe exact `rg`-style search remains necessary. Decide whether MCP should expose an indexed exact tool, only hand off to host `rg`, or both.

### 12. Implementation roadmap

Give phased implementation slices that can become specs.

Expected rough direction:

- P4.A: Qdrant sparse vectors + RRF;
- P4.B: code-aware lexical normalization;
- P4.C: payload indexes / Qdrant filters;
- P4.D: context search response route;
- P5.A: graph artifact + graph status;
- P5.B: seed binding + 1-hop expansion;
- P5.C: evidence paths + graph diagnostics;
- P5.D: better edges/imports/tests/docs/config;
- P6+: tree-sitter, route/migration/db edges, typed PPR, ColBERT/rerank.

Correct this sequence if needed.

### 13. What not to build

Explicitly list anti-goals:

- hidden ReAct agent;
- hidden LLM query planner;
- OpenIE-first graph;
- raw graph dumps;
- GraphRAG summary engine as primary search;
- automatic lifecycle mutation;
- over-broad graph expansion;
- anything else you think is harmful.

### 14. Risks and mitigations

Cover:

- retrieval quality risk;
- stale graph risk;
- latency risk;
- context bloat;
- language coverage gaps;
- parser failures;
- generated/vendor files;
- score/fusion instability;
- agent misuse.

### 15. Final decision table

End with a decision table:

```text
Decision | Recommendation | Why | Spec target
```

### 16. Open questions

List only real remaining questions that must be settled before implementation specs.

---

## Tone and quality bar

Be concrete. Avoid generic RAG advice.

This project already has a serious dense+sparse retrieval baseline and a good HippoRAG research pass. Your job is to synthesize those into a pragmatic architecture for a real MCP that coding agents will use every day.

The final output should be detailed enough that we can split it into implementation specs without asking another model to "think harder" about the architecture.
```
