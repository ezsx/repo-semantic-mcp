# P5 Deep Research Prompt: HippoRAG for `repo-semantic-search`

# Deep Research: HippoRAG / graph-augmented retrieval for code-repository MCP search

## Role

You are a senior RAG architect and research engineer. Your task is to deeply research HippoRAG and adjacent graph-augmented retrieval techniques, then evaluate how they should shape the next architecture of `repo-semantic-search`: an MCP server used by frontier coding agents to work in large software repositories.

Do not answer as a generic literature survey. Treat this as an architecture research task for a real tool that will be implemented. Use current papers, implementations, open-source repos, and production-oriented RAG design knowledge. Challenge our assumptions where needed.

---

## Project context

`repo-semantic-search` is an MCP server for repository search. It is intended for frontier models like GPT-5.x, Claude, Gemini, etc. The target user is not a human typing into a chatbot UI; the target user is a coding agent that needs to gather context quickly and correctly before editing a large repository.

The project goal:

- simplify work for frontier models in large codebases;
- improve fast and correct context gathering;
- make repository search safer and more ergonomic for agents;
- return grounded evidence with paths, line ranges, snippets, warnings, and diagnostics;
- avoid agents wasting time opening many irrelevant files;
- preserve exact-search workflows for symbols, env vars, route names, SQL names, errors, and literals;
- keep lifecycle/index operations explicit and safe.

This is not meant to become a standalone RAG chatbot. It is a retrieval layer for daily software development.

The frontier model remains the planner and final reasoning engine. The MCP should provide strong retrieval primitives, good diagnostics, and grounded results. It should not hide a slow internal autonomous agent, full ReAct loop, final-answer generator, or LLM query planner unless there is a very strong reason.

---

## Current `repo-semantic-search` state

The MCP already has:

- Qdrant-backed persistent index;
- separate logical `code` and `docs` collections;
- chunking for Python, Markdown, JSON/YAML/TOML, and generic text;
- chunks with `chunk_id`, `relative_path`, `scope`, `language`, `chunk_type`, `start_line`, `end_line`, `symbol_path`, `heading_path`, `domain_tags`, `is_generated`;
- `index_status.v2` with repo identity, active repo, branch/commit freshness, dirty/untracked info, backend/runtime checks, lifecycle state, compatibility contracts, and host-side hints;
- `search_v2` with path filters, include/exclude filters, extension/language/chunk_type/domain filters, snippets, matched terms, line matches, warnings, diagnostics;
- semantic search and hybrid search;
- explicit lifecycle boundary: the agent should not silently rebuild/reindex/switch runtime; status must explain what host-side action is needed.

The main weakness is retrieval quality and architecture depth:

- semantic search is mostly "embed chunks and retrieve dense nearest neighbors";
- hybrid search currently scrolls indexed chunks into an in-process BM25 cache;
- dense and lexical scores are blended with a fixed score formula;
- Qdrant does not yet store a real sparse lexical vector branch for repo chunks;
- there is no graph layer connecting files/chunks/symbols/tests/docs/config;
- agents still need to manually infer neighboring context after a seed result.

---

## Baseline we already believe in from a separate RAG project

We have a separate recent RAG project over a real corpus. It used modern retrieval components and produced fixed, measured quality improvements. We do not want to rerun a long ablation project right now, but we trust the main lessons from that system.

That project used:

- Qdrant named vectors;
- dense embeddings;
- sparse/BM25 recall;
- weighted RRF fusion;
- ColBERT late interaction / MaxSim;
- cross-encoder rerank/filter;
- sparse-only lexical normalization;
- original-query injection into every multi-query search;
- multi-query retrieval and merge;
- grounded context/citation envelopes;
- coverage/refinement diagnostics;
- payload indexes and metadata filters.

Important lessons from that project:

1. Dense-only retrieval is not enough.
2. Lexical/BM25 recall is critical for exact anchors.
3. Dense input should usually stay raw.
4. Lexical/sparse input can be normalized separately.
5. RRF is safer than blending unrelated score scales.
6. Original query must always be preserved; LLM rewrites can erase lexical anchors.
7. ColBERT and rerank are valuable but should be explicit optional stages.
8. Grounding and diagnostics are as important for agents as raw ranking.
9. The agent should remain outside the retrieval engine; the retriever should expose evidence and route diagnostics.

For `repo-semantic-search`, the planned non-graph baseline is:

```
agent original query / optional subqueries
  -> dense branch: raw query
  -> sparse branch: code-aware lexical normalization
  -> Qdrant dense recall + Qdrant sparse recall
  -> weighted RRF fusion
  -> optional rerank / optional ColBERT later
  -> grounded SearchResponse with path/line snippets, matched terms, branch diagnostics
  -> agent reads files/chunks and verifies exact anchors when needed
```

Code-aware lexical normalization should preserve and split useful code anchors:

- snake_case;
- camelCase;
- kebab-case;
- dotted module paths;
- slash paths;
- HTTP routes;
- env vars;
- symbols;
- filenames;
- SQL table names;
- migration names;
- CLI flags;
- error messages;
- config keys.

We do not want to put a slow LLM planner inside MCP by default. The frontier model can plan subqueries. MCP may accept `query` plus `subqueries`, and may provide deterministic recipes/routes, but should not become another hidden ReAct system.

---

## Why HippoRAG is interesting for this project

We think HippoRAG-style graph retrieval may be especially useful for code repositories because code is naturally connected.

In ordinary text RAG, a chunk may be relevant because it is semantically similar to the query. In code search, the most useful context is often related by structure rather than raw similarity:

- endpoint implementation -> service function -> repository/db layer -> schema/migration;
- function -> callers/callees;
- class -> methods -> tests;
- route -> API docs/spec -> request/response schema -> integration tests;
- env var -> config loader -> docker compose -> deployment docs;
- SQL table -> model/schema -> migration -> repository methods;
- bot handler -> frontend flow -> backend API -> docs/spec;
- agent policy file -> module-specific context -> implementation area;
- failing test -> tested module -> fixture -> config.

A dense+sparse search seed may find one strong file/chunk, but the useful development context may live one or two graph hops away. This is the central hypothesis to validate.

We do not want graph retrieval to replace dense+sparse retrieval. We currently imagine graph retrieval as an expansion layer over high-quality seeds:

```
dense+sparse+RRF seeds
  -> graph expansion over code/documentation relationships
  -> edge/path-aware scoring
  -> fusion/rerank with original candidates
  -> grounded response
```

But this is only a hypothesis. Please research whether this is the right mental model.

---

## What to research about HippoRAG

Research HippoRAG deeply:

1. What exactly is HippoRAG?
2. What papers introduced it? Are there HippoRAG v1/v2 variants?
3. What problem does it solve compared to dense retrieval, sparse retrieval, GraphRAG, LightRAG, RAPTOR, or ordinary knowledge graph RAG?
4. How does it build graph nodes and edges from a corpus?
5. Does it use extracted entities, OpenIE triples, chunk nodes, concept nodes, document nodes, or other structures?
6. How does query-time retrieval work?
7. How are seeds selected?
8. How does graph traversal/expansion happen?
9. What ranking/scoring/fusion method is used after graph expansion?
10. Does it use personalized PageRank, path scoring, embedding similarity, graph centrality, LLM extraction, or another mechanism?
11. What are the measured benefits and on which tasks?
12. What are its failure modes and operational costs?
13. What implementation details matter most?
14. Which ideas are essential and which are incidental to the original benchmark setup?

Use primary sources where possible: papers, official repos, author docs, implementation code, technical reports. Compare against adjacent systems only when useful for architecture decisions.

---

## Main architecture questions for code search

Answer these as directly as possible.

### 1. Is our hypothesis correct?

Is HippoRAG-style graph expansion a good fit for repository search? Is code structure a stronger and cleaner graph signal than entity graphs in open-domain text? Where is the analogy valid, and where does it break?

We need a judgment, not just "it depends".

### 2. Where should graph retrieval sit in the route?

Evaluate these options:

Option A:
```
dense+sparse retrieval -> RRF -> top seeds -> graph expansion -> final fusion
```

Option B:
```
graph search and dense+sparse search run as parallel branches -> RRF/fusion
```

Option C:
```
query intent router chooses either direct dense+sparse route or graph route
```

Option D:
```
graph expansion only after the agent asks for "related context" / "cross-file flow"
```

Which route is most practical for an MCP used by frontier coding agents?

### 3. What graph should we build?

Propose a code-repo graph schema.

Candidate node types:

- repository;
- file;
- chunk;
- symbol;
- function;
- method;
- class;
- module/package;
- import target;
- route/endpoint;
- config key;
- env var;
- SQL table;
- migration;
- test;
- fixture;
- document/spec section;
- agent policy/context module.

Candidate edge types:

- file contains chunk;
- chunk defines symbol;
- class contains method;
- function calls function;
- file imports module;
- test targets symbol/file;
- fixture used by test;
- docs/spec references endpoint/symbol/config;
- route handled by function;
- config key read by code;
- env var defined in compose/deploy and read by config;
- migration creates/alters table;
- model/repository uses table;
- package depends on package;
- agent policy applies to path/module.

For each edge type, classify:

- high-signal and cheap for v1;
- useful but parser/LSP-dependent;
- too noisy or expensive for v1;
- language-specific;
- requires later enrichment.

### 4. How should graph edges be built?

Evaluate practical extraction methods:

- regex/token heuristics;
- Python AST;
- tree-sitter for multi-language support;
- static import graph;
- call graph approximation;
- LSP references;
- ripgrep/exact reference scans;
- config/env parsers;
- docs link/reference extraction;
- test naming conventions;
- package manager metadata.

We need a pragmatic v1. This MCP should work across repositories, not only one Python codebase. But it can start with a minimal portable graph and grow.

### 5. How should graph retrieval be scored?

Research and recommend ranking/fusion methods:

- RRF between dense, sparse, graph branches;
- personalized PageRank from seed nodes;
- path-length penalty;
- edge-type weights;
- node-type priors;
- query intent weights;
- rerank after graph expansion;
- ColBERT after graph expansion;
- exact anchor boost;
- diversity by file/scope/node type.

Important: do not blend incomparable raw scores naively. Our prior experience favors rank-based fusion.

### 6. How to avoid graph noise?

Code graphs can explode. Imports, common utils, base classes, fixtures, and config files may connect everything.

Research safeguards:

- edge-type allowlists by route;
- max hops;
- max neighbors per seed;
- high-degree node penalties;
- generated/vendor/test fixture controls;
- path/domain filters;
- graph expansion budget;
- explainable paths in response;
- exact-anchor gating before expansion;
- reranker/fusion after expansion.

### 7. How should freshness/staleness work?

This MCP already cares about branch/worktree/index freshness. Graph retrieval adds another stale artifact.

Research and propose:

- graph schema version;
- graph built commit;
- per-file content hash;
- incremental graph update when files change;
- invalidating edges for changed files;
- handling untracked files;
- graph status in `index_status`;
- behavior when vector index is fresh but graph is stale;
- behavior when graph is partially available.

The agent must not silently trust stale graph context for code edits.

### 8. What should the response shape expose?

The output is consumed by a frontier coding agent.

Recommend additions to `SearchResponse` / future `repo_context_search`:

- route used;
- queries used;
- dense/sparse/graph candidate counts;
- graph seed nodes;
- graph-expanded nodes;
- edge paths explaining why a result was included;
- node/edge types;
- branch-specific ranks/scores;
- final fused rank;
- warnings about stale graph or best-effort extraction;
- recommended next actions: `read_chunk`, file read, exact search, inspect tests, inspect docs/spec.

The response must stay compact enough to be useful in an MCP call.

---

## Important constraints

This project is for daily engineering use, not a paper reproduction.

Constraints:

- avoid long ablation plans as the main output;
- avoid requiring a cloud API;
- avoid requiring an LLM call for every search;
- avoid making graph extraction so heavy that onboarding a repo becomes painful;
- preserve exact search fallback;
- preserve existing status/lifecycle safety;
- old search tools must remain compatible;
- graph layer should degrade gracefully when unavailable;
- implementation should be sliceable.

Reasonable stack assumptions:

- Python;
- Qdrant;
- local embedding/reranker backends;
- optional ColBERT later;
- local filesystem/git access;
- likely tree-sitter or AST parsers later if justified;
- MCP clients are coding agents, not end-user chat UI.

---

## Desired output format

Return a structured research report with these sections:

### 1. Executive summary

- Is HippoRAG relevant to code-repo search?
- What should we borrow?
- What should we avoid?
- What is the recommended high-level route?

### 2. HippoRAG explanation

Explain HippoRAG clearly:

- core algorithm;
- graph construction;
- query-time retrieval;
- ranking/fusion;
- measured strengths;
- limitations.

### 3. Comparison to adjacent approaches

Briefly compare:

- HippoRAG;
- GraphRAG;
- LightRAG;
- RAPTOR;
- dense+sparse RAG;
- code-specific graph/static-analysis retrieval if relevant.

Focus only on architecture implications.

### 4. Fit analysis for code repositories

Evaluate our hypothesis:

- where graph retrieval should help;
- where it will not help;
- which code tasks benefit most;
- which query types should bypass graph expansion.

### 5. Proposed architecture sketch

Give a concrete route, for example:

```
query/subqueries
  -> dense raw retrieval
  -> sparse code-normalized retrieval
  -> RRF seed fusion
  -> graph expansion from seed nodes
  -> graph candidate scoring
  -> final rank fusion / optional rerank
  -> grounded SearchResponse
```

If a different route is better, propose it and explain why.

### 6. Code graph schema

Provide:

- node types;
- edge types;
- payload fields;
- freshness fields;
- schema versioning;
- high-signal v1 subset;
- later extensions.

### 7. Retrieval and scoring details

Recommend:

- seed selection;
- max hops;
- edge weights;
- high-degree penalties;
- fusion method;
- rerank placement;
- diversity constraints;
- compact explanation format.

### 8. Implementation phases

Give a pragmatic implementation plan:

Phase 1 should be small and useful. It might be:

- build file/chunk/symbol/import/test/doc/config graph subset;
- add graph status;
- add graph expansion over dense+sparse seeds;
- expose graph paths in response.

Then later phases can add:

- stronger static analysis;
- LSP/tree-sitter;
- call graph;
- docs-to-code links;
- ColBERT/rerank integration;
- graph-aware recipes.

### 9. Risks and mitigations

Include:

- noisy graph expansion;
- stale edges;
- latency;
- over-context;
- generated/vendor files;
- language coverage gaps;
- false confidence from weak edges;
- operational complexity.

### 10. Open questions for final architecture

List the remaining decisions we must make before implementation.

---

## Final instruction

Be opinionated. We are not asking whether graph RAG is interesting in general. We need to decide how to build the next `repo-semantic-search` architecture. Validate or correct our mental model, then give a practical architecture sketch that can be handed to another frontier model for final architecture design.
