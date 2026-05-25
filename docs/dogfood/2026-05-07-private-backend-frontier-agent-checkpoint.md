# Dogfood Checkpoint: Private Backend Frontier-Agent Retrieval

Date: 2026-05-07

Target repository:

```text
Large private backend/control-plane repository
```

Runtime profile:

```text
gpu-pplx-v100-infinity
```

Observed status:

```text
search_available: true
runtime_switch_required: false
backend_switch_required: false
index_stale: false
indexed chunks: 16966
code chunks: 7806
docs chunks: 9160
graph state: ready
graph nodes: 36097
graph edges: 48862
```

The target repo worktree was dirty during the test, but the index was not stale
under the mtime-aware freshness policy because the changed indexable files were
not newer than the successful index update. Exact local verification remains
recommended before edits.

## Goal

Validate whether `repo-semantic-search` is useful to a frontier coding agent in
a large, real backend/control-plane repository.

The useful behavior is not "find some files". The useful behavior is:

- identify whether the index is usable;
- gather source-of-truth context before editing;
- connect docs, code, tests, deploy, and agent policy;
- use graph expansion for cross-area flows;
- route exact-anchor tasks back to local `rg`;
- avoid hidden lifecycle operations.

## Scenario 1: Project Overview

Prompt shape:

```text
Use MCP-first. Call index_status, verify repo_root/search_available/freshness,
do not call lifecycle tools, then gather a human-readable map of the private
backend/control-plane repo: main runtime areas, frontend/user entrypoints,
API services, worker services, deploy/compose, docs/specs/testing, and agent
policy.
```

Result:

- The agent started with `index_status`.
- It identified the project as a backend/control-plane system.
- It found the main runtime contours:
  - user-facing issuance flow;
  - frontend/user entrypoints;
  - API services;
  - maintenance workers;
  - deploy/compose;
  - docs/specs/testing;
  - repo-owned agent context.
- It explained the core user path in sanitized form:

```text
User entrypoint / frontend action
  -> API facade
  -> issuance flow
  -> job state
  -> job queue
  -> worker
  -> ready generated artifact
```

Product signal:

The answer moved beyond a raw file list and produced a mental model that a new
agent could use before editing.

## Scenario 2: Cross-Area Implementation Map

Prompt shape:

```text
Find the full implementation context for a representative backend route.
Explain endpoint, handler, job state, queue enqueue, worker, response mapping,
docs/specs, tests, and deploy/env dependencies. Use MCP-first and local rg only
for exact verification anchors.
```

Key file groups found:

- API endpoint declaration and runtime dependencies.
- API auth/identity boundary.
- Facade over the user-facing issuance flow.
- Handler logic: deterministic job id, pending state, queue enqueue, wait loop,
  ready/pending response.
- Job identity helper.
- Job-state storage contract.
- Worker registration.
- Worker implementation for artifact generation.
- Frontend/client call into the backend route.
- Frontend operation and artifact delivery path.
- Deploy/compose runtime env/service wiring.
- Product/spec docs that describe the route contract.
- Unit and integration test surface.

Product signal:

The agent produced an implementation map that crossed services, docs, tests, and
deploy files without mixing the target flow with adjacent maintenance contours.

## Scenario 3: Graph Value

Prompt shape:

```text
Use repo_context_search with graph_mode=expand. Trace the connection from a
frontend connect action to backend issuance. Report effective graph mode,
branches, evidence paths, and where graph helped compared to grep.
```

Observed behavior:

- `graph_mode_effective=expand`.
- `graph_used=true`.
- Relevant searches returned dense, sparse, and graph branches.

Important connected path, sanitized:

```text
frontend action
  -> frontend runtime operation
  -> API client
  -> backend route
  -> API endpoint
  -> facade
  -> handler
  -> job queue enqueue
  -> worker registration
  -> worker implementation
  -> ready/failed state
  -> frontend artifact delivery
```

Docs connected by retrieval:

- Observability/spec document containing an end-to-end trace from frontend
  action through API, issuance flow, job queue, worker, and artifact store.
- Product/spec document connecting frontend behavior, operation state, and
  artifact state.

Product signal:

The graph branch was valuable for cross-area explanation. Plain `rg` could find
literal strings, but it would return islands. The graph-assisted route helped
explain why the islands belong to one end-to-end flow.

## Scenario 4: Exact-Anchor Handoff

Prompt shape:

```text
Find only all uses of a specific env/config secret. Start MCP-first, but base
final confidence on exact rg. Do not expand graph if this is a simple exact
lookup.
```

Observed behavior:

- MCP classified the query as exact-anchor-heavy.
- It recommended exact local verification.
- `graph_mode_effective=off`.
- Authoritative command shape:

```powershell
rg --fixed-strings --line-number --no-heading -e <ENV_OR_CONFIG_ANCHOR> .
```

Result categories:

- deploy/env definitions;
- runtime reads;
- docs/spec mentions;
- tests;
- unrelated mentions.

Practical finding in the target repo:

A sample env file did not contain a required env/config anchor while production
compose/docs expected it. This may be a real follow-up finding for the target
repository, outside this MCP checkpoint.

Product signal:

The MCP did not overuse graph search. It acted as a routing layer and correctly
handed exact verification to local `rg`.

## Acceptance Notes

This checkpoint supports the current architecture direction:

- dense+sparse retrieval is useful as the baseline;
- graph expansion is useful for connected repository context;
- exact local search remains necessary and authoritative;
- lifecycle operations must remain explicit;
- agent-facing diagnostics make the tool safer to use in large repositories.

The next product polish area is evidence presentation. Agents can currently
explain evidence paths, but the MCP should keep making `origin_branches`,
`graph_mode_effective`, and `evidence_paths` more stable and easier to quote in
human-readable answers.
