# P6: Final architecture artifact index

This is the upload/read order for the GPT-5.5 Pro architecture pass.

## Required artifacts

### 1. Current agent-grade MCP contract

File:

- `docs/specifications/P3-agent-grade-status-search-and-exact-fallback-specification.md`

Why it matters:

- captures the current status/search/exact-fallback direction;
- defines `index_status.v2`, `search_v2`, lifecycle safety, filters, snippets,
  diagnostics, and agent ergonomics;
- prevents the final architecture from breaking already accepted MCP contracts.

### 2. Transfer plan from the existing RAG project

File:

- `docs/research/rag-transfer/P4-rag-stack-transfer-from-rag-app.md`

Why it matters:

- defines what we take from the separate proven `rag_app` stack;
- fixes the dense+sparse+RRF baseline direction;
- clarifies that we transfer retrieval layers, not the full ReAct app;
- lists first implementation slices for sparse vectors, lexical normalization,
  filters, context search, and optional rerank/ColBERT.

### 3. HippoRAG deep research report

File to upload manually:

- `<downloaded-deep-research-report.md>`

Why it matters:

- external research result on HippoRAG and graph retrieval;
- compares HippoRAG, GraphRAG, LightRAG, RAPTOR, RepoHyper, RepoGraph,
  CodexGraph;
- argues for seeded graph expansion over dense+sparse seeds;
- proposes graph schema, graph scoring, freshness, and response shape.

### 4. Our analysis of the HippoRAG report

File:

- `docs/research/hipporag/P5-hipporag-research-analysis.md`

Why it matters:

- converts the research report into project decisions;
- separates accepted conclusions from scope corrections;
- proposes storage, graph tables, query route, graph scoring, implementation
  sequence, and open decisions.

## Optional artifacts if context budget allows

### Current repo overview

Files:

- `README.md`
- `docs/specifications/P1.Z-repo-semantic-search-mcp-specification.md`
- `docs/specifications/P2-repo-semantic-search-lifecycle-and-backend-abstraction-specification.md`

Why:

- useful for historical context, but less important than P3-P5.

### Current implementation files

Files:

- `services/repo_semantic/models.py`
- `services/repo_semantic/search_service.py`
- `services/repo_semantic/qdrant_store.py`
- `services/repo_semantic/indexer.py`
- `services/repo_semantic/chunkers/factory.py`
- `services/repo_semantic/mcp_server.py`

Why:

- useful if GPT-5.5 Pro should produce architecture that directly maps to
  current modules;
- not required if the task is architecture only.

## Recommended read order

1. Read the GPT-5.5 prompt in `docs/research/final-architecture/P6-final-architecture-gpt55-prompt.md`.
2. Read P3 to understand the current MCP contract and safety boundaries.
3. Read P4 to understand the proven RAG stack transfer.
4. Read the external HippoRAG deep research report.
5. Read P5 analysis to see which parts of the report we accept or constrain.
6. Optionally inspect README/P1/P2/current code if more grounding is needed.

## Expected result from GPT-5.5 Pro

We want a final architecture and development vector, not implementation code.

The output should be good enough to turn into several implementation
specifications:

- P4 implementation specs for dense+sparse Qdrant retrieval and RRF;
- P5 implementation specs for graph artifact and seeded graph expansion;
- P6 implementation specs for advanced rerank/ColBERT/tree-sitter/PPR phases;
- API/status/model contracts for future `search_v3` or `repo_context_search`.

## Final architecture output

The GPT-5.5 Pro architecture result has been normalized into:

- `docs/architecture/P6-final-architecture.md`

Use it as the source-of-truth architecture baseline before writing the next
implementation specifications.
