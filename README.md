# repo-semantic-mcp

`repo-semantic-mcp` is a repository retrieval MCP for frontier coding agents.
It helps an agent gather grounded context in a large codebase before editing:

- check whether the index is ready and fresh enough to use;
- retrieve relevant code, docs, tests, deploy files, and policy files;
- combine dense semantic search with code-aware sparse retrieval;
- optionally expand strong results through a typed repository graph;
- tell the agent when exact local `rg` verification is the right authority.

The MCP is not a hidden coding agent and does not generate final answers. The
frontier model remains the planner, reader, editor, and reviewer. This server is
the retrieval substrate underneath it.

## Current Shape

The current implementation provides:

- Qdrant-backed `code` and `docs` collections;
- named dense and sparse vectors, with weighted RRF fusion;
- code-aware lexical normalization for identifiers, paths, routes, env vars, and
  quoted literals;
- `index_status.v2` with runtime, backend, freshness, retrieval, and graph
  diagnostics;
- `repo_context_search` for agent-facing context envelopes;
- `search_v2` plus legacy list-shaped semantic/hybrid tools;
- SQLite graph artifact with explicit `build_graph` / `rebuild_graph` /
  `update_graph`;
- bounded graph expansion for cross-file repository context;
- watcher startup reconcile and bounded graph recovery after branch switches,
  restarts, and incremental index updates;
- explicit exact-search handoff instead of shelling out inside the MCP.

The accepted long-term architecture is documented in
`docs/architecture/P6-final-architecture.md`.

## Why It Exists

Large repositories are hard for coding agents because useful context is rarely
in one file. A change may touch:

- runtime code;
- tests;
- deployment config;
- docs/specifications;
- agent policy files;
- env contracts;
- cross-service flows.

Plain `rg` is still the authority for exact identifiers and literals, but it
does not explain which distant files belong to the same implementation flow.
Semantic search helps with concepts, but raw embeddings alone often miss exact
anchors and code structure. This project combines both and adds a small,
deterministic graph layer for connected repository context.

## Where It Adds Value

The goal is not "semantic search instead of grep". The goal is to give a
frontier coding agent a better first map of a large repository.

`rg` is excellent when the agent already knows the exact string. It is still
the authority for identifiers, env vars, route literals, table names, and error
messages in the live working tree.

This MCP is useful when the agent needs to answer questions like:

- what files belong to this feature or runtime flow?
- which docs/specs/tests/deploy files explain the code I am about to edit?
- how does a route, env var, config key, or policy connect across the repo?
- after a restart or branch switch, can I trust the index and graph enough to
  use them before editing?

Practical examples from private-repository dogfood:

- A query about a frontend-to-backend connect flow linked product docs,
  observability specs, API route code, facade code, and worker context. Plain
  exact route lookup found dozens of matches, but
  did not explain the end-to-end route.
- A query about user traffic accounting found the maintenance telemetry SQL,
  handler code, and relevant unit/integration tests around
  hashed peer identifiers and hourly traffic tables. Exact `rg` remained useful
  afterward for verification.
- Exact env var lookup still routes to local `rg`; the MCP contributes by
  detecting that this is an exact-anchor task and by exposing freshness state.

## Retrieval Route

The main route is:

```text
agent query / agent-provided subqueries
  -> dense retrieval over Qdrant
  -> sparse code-normalized retrieval over Qdrant
  -> weighted RRF fusion
  -> optional typed graph expansion from strong seeds
  -> final RRF over dense+sparse+graph branches
  -> grounded context envelope
```

Graph expansion is useful for cross-area questions such as:

- "where is this flow implemented end to end?"
- "find handler, tests, docs, and deploy contract for this route"
- "connect this frontend action to the backend worker"
- "where does this env/config contract move through the system?"

Graph expansion is intentionally suppressed or avoided for plain exact lookups.
For exact env vars, route strings, table names, error messages, and path
fragments, the MCP should recommend local `rg` verification.

## Important Tools

Agent-facing retrieval:

- `repo_context_search` - primary context route with branch diagnostics,
  file groups, matched/uncovered terms, graph diagnostics, and next actions.
- `search_v2` - stable search envelope without mandatory graph semantics.
- `semantic_search`, `semantic_search_code`, `semantic_search_docs` - legacy
  semantic result lists.
- `hybrid_search`, `hybrid_search_code`, `hybrid_search_docs` - legacy hybrid
  result lists.

Readiness and lifecycle:

- `index_status` - first tool an agent should call.
- `graph_status` - detailed graph state.
- `build_index`, `rebuild_index`, `reindex_paths` - explicit index lifecycle.
- `build_graph`, `rebuild_graph`, `update_graph` - explicit graph lifecycle.
- `start_watcher`, `stop_watcher` - explicit watcher lifecycle.

Search and status never rebuild indexes or graphs implicitly.

## Agent Workflow

Recommended policy for a coding agent working in a target repository:

1. Call `repo-semantic-search.index_status()`.
2. Verify `repo_root` is the intended repository.
3. Verify `search_available=true`.
4. Check runtime/backend/freshness/graph diagnostics.
5. Use `repo_context_search` or `search_v2` to narrow scope.
6. Read the returned files/ranges.
7. Use local `rg` for exact anchors before editing.
8. Do not call lifecycle tools unless the user explicitly asks, or unless the
   task is an explicitly permitted safe-recovery flow.

If the MCP reports `runtime_switch_required`, `backend_switch_required`, or a
blocking lifecycle state, the agent should stop and report the host-side hint
instead of trying to fix it silently.

For day-to-day agent recovery, prefer repair-first operations over rebuilds:

```powershell
py -3.12 scripts/agents/repo_semantic_ensure_ready.py `
  --repo C:\path\to\target-repo `
  --mode safe-recover `
  --start-watcher `
  --require-graph `
  --json
```

This may start the watcher and run bounded startup reconcile for the already
active matching repo/profile. It may also run bounded graph repair when the
index is fresh and graph recovery is safe. It still does not build or rebuild
the index or graph.

## Runtime Model

One running stack indexes one target repository at a time. The target repository
is an explicit launch parameter; the MCP does not guess the active IDE folder.

This is deliberate:

- Codex and Claude can share the same MCP endpoint;
- target repo switching is deterministic;
- collection names include repo/profile/model/schema identity;
- lifecycle operations stay explicit.

## Quick Start

Register the agent integrations once:

```powershell
pwsh -File scripts/agents/register_repo_semantic_search.ps1
```

Start the stack for a target repository:

```powershell
pwsh -File scripts/agents/start_repo_semantic_for_project.ps1 `
  -RepoPath C:\path\to\target-repo
```

For an explicit profile:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

CPU fallback:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile cpu `
  -TargetRepoPath C:\path\to\target-repo
```

V100 / Volta profile used during current dogfood:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu-pplx-v100-infinity `
  -TargetRepoPath C:\path\to\target-repo
```

After registration, restart the client if it was already open. Codex uses a
host-side stdio wrapper that forwards to the shared HTTP endpoint, so backend
container restarts do not kill the chat transport. Claude uses the same shared
HTTP endpoint directly.

## Profiles

The stack is backend-abstracted: the MCP talks to configured endpoints for
embeddings, and future reranker/ColBERT endpoints are modeled the same way. The
server should not care whether a model runs in WSL native, llama.cpp native,
Docker GPU, or another host process, as long as the endpoint contract matches.

Common profiles:

- `cpu` - compatibility profile using `intfloat/multilingual-e5-small`.
- `gpu` / `gpu-qwen3` - NVIDIA Docker profile for Qwen3 embeddings.
- `gpu-bge-m3` - experimental/debug profile.
- `gpu-pplx-v100-infinity` - V100-friendly Infinity profile for
  `pplx-embed-v1`.

Qdrant defaults to `qdrant/qdrant:v1.17.1`; current index schema is `3`.

## Readiness

The source of truth is:

```powershell
pwsh -File scripts/agents/repo_semantic_status.ps1
```

or:

```text
http://127.0.0.1:8011/readyz
```

`/readyz` reports runtime readiness, active repo, profile, index state,
`search_available`, and embedded `index_status` details. Docker container
health is useful, but `/readyz` and `index_status` are the operational checks
agents should rely on.

For agent-side preflight:

```powershell
py -3.12 scripts/agents/repo_semantic_ensure_ready.py --repo C:\path\to\target-repo
```

The helper is read-only by default (`diagnose` mode). It reports whether search,
sparse retrieval, graph expansion, watcher, and HTTP fallback are usable. To let
an agent perform bounded repair, use explicit `--mode safe-recover`; this may
start the watcher for the already active matching repo/profile, but still never
builds or rebuilds the index or graph.

For direct HTTP fallback tool calls:

```powershell
py -3.12 scripts/agents/repo_semantic_call_tool.py index_status
```

That script prints a stable `ok/error` envelope by default. Use `--payload` only
when a human specifically wants the decoded MCP payload without wrapper fields.

For a broader reliability smoke against a target repo:

```powershell
py -3.12 scripts/agents/repo_semantic_reliability_smoke.py `
  --repo C:\path\to\target-repo
```

The smoke uses HTTP fallback, validates readiness, runs a tiny
`repo_context_search` probe, checks `graph_status`, and reports stable scenario
results. It is no-rebuild by default: it does not build indexes, rebuild graphs,
switch repos/backends, or mutate source files. Add `--allow-start-watcher` only
when the agent is explicitly allowed to start the watcher for the already active
repo.

Watcher status is explicit:

- `watcher.enabled=true` means the runtime is configured to support incremental
  watching.
- `watcher.running=true` means a polling watcher is actually active.
- `watcher.start_required=true` means live incremental indexing will not happen
  until `start_watcher` is called explicitly.
- `watcher.autostart_policy=manual` is intentional for now; status/search tools
  do not start lifecycle operations by themselves.

When called explicitly, `start_watcher` first runs a startup reconcile for files
changed while the watcher was stopped, then starts polling for future changes.
It is still an explicit lifecycle operation, not something status/search calls do
implicitly.

Graph freshness is maintained separately from vector freshness. After P6.B,
bounded recovery can update the graph from the current indexed chunks without a
full graph rebuild. This is important after:

- branch switches;
- MCP/container restarts;
- watcher catch-up that changed indexed chunks;
- legacy graph artifacts missing per-file input signatures.

If graph recovery exceeds safe automatic bounds, status should recommend
permission-required `update_graph(allow_large_update=true)` before recommending
`rebuild_graph`. Rebuild remains the last resort for hard artifact/schema
mismatches or genuinely unbounded updates.

## Dogfood Checkpoint

The current checkpoint was dogfooded against a large private
backend/control-plane repository:

- profile: `gpu-pplx-v100-infinity`;
- indexed chunks: `17035`;
- graph: ready, `36250` nodes and `49015` edges;
- watcher: running;
- graph expansion: allowed and verified through `repo_context_search`.

Practical scenarios passed:

1. project overview for a new coding agent;
2. cross-area implementation map for a representative backend route;
3. exact-anchor lookup for an env/config secret with `rg` authority and graph
   suppressed.
4. graph-assisted frontend/backend flow context search with
   `graph_mode=expand` and `graph_used=true`;
5. safe-recovery after MCP restart without index or graph rebuild;
6. small-fixture branch-switch e2e:
   `start_watcher` -> startup reconcile -> graph update ->
   graph search returns the fresh route.

The older checkpoint is kept in
`docs/dogfood/2026-05-07-private-backend-frontier-agent-checkpoint.md`; the README
reflects the newer P6.B recovery dogfood state.

## Current Limitations

- The stack still requires deliberate setup. It is not yet a one-command,
  polished community install.
- Direct MCP transport can be client-dependent; the HTTP fallback wrapper is the
  reliable path used by current agent scripts.
- Sparse BM25 stats may become stale after incremental changes. Hybrid search
  remains usable, but exact anchors should still be checked with local `rg`.
- Graph extraction is useful but intentionally bounded. It is not a full
  call-graph engine, not GraphRAG summarization, and not a hidden planner.
- Some graph extractor warnings are nonblocking today, especially for Python AST
  parse edge cases in large real repositories.

## Repository Layout

- `apps/repo-semantic-mcp/` - MCP app entrypoint and container image.
- `services/repo_semantic/` - indexing, retrieval, status, graph, storage, and
  backend logic.
- `deploy/repo-semantic-search/` - Docker Compose stack and env profiles.
- `scripts/agents/` - startup, registration, status, and client wrappers.
- `docs/architecture/` - accepted architecture.
- `docs/specifications/` - implementation specs.
- `docs/research/` - research inputs and prompts.
- `docs/dogfood/` - practical dogfood checkpoints.
- `tests/` - unit and integration coverage for the MCP service.

## Development Notes

- Legacy list-shaped tools must remain backward-compatible.
- Lifecycle tools must remain explicit.
- Dense query text stays raw.
- Sparse query text is code-normalized while preserving exact anchors.
- Weighted RRF is used instead of direct raw score blending.
- Graph expansion is bounded, deterministic, and explainable.
- Stale graph/index state must be visible and must not silently affect results.
- Exact local `rg` remains authoritative when the agent has filesystem access.

## License

This repository is prepared for publication under `PolyForm Noncommercial
1.0.0`.

That means copying and modification are allowed under the license terms, but
commercial use is not allowed unless the license is changed explicitly.
