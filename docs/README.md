# Documentation map

This repository keeps research inputs, accepted architecture, and implementation
specifications separate. When starting new work, read from the top of this map
instead of scanning every file in `docs/`.

## Source of truth for next work

Read first:

- `architecture/P6-final-architecture.md`
- `dogfood/2026-05-07-private-backend-frontier-agent-checkpoint.md`

This is the accepted architecture baseline for the next implementation specs.
It defines the future route:

```text
dense raw retrieval
  -> sparse code-normalized retrieval
  -> RRF seed fusion
  -> optional seeded graph expansion
  -> grounded repo context envelope
```

The dogfood checkpoint is the current practical proof that the architecture is
useful to a frontier coding agent in a large real repository. It should be used
as a regression scenario when retrieval, graph expansion, freshness, or exact
handoff behavior changes.

## Specifications

Implementation contracts and accepted product specs:

- `specifications/P1.Z-repo-semantic-search-mcp-specification.md`
- `specifications/P2-repo-semantic-search-lifecycle-and-backend-abstraction-specification.md`
- `specifications/P3-agent-grade-status-search-and-exact-fallback-specification.md`
- `specifications/P4.A-qdrant-sparse-vectors-and-rrf.md`
- `specifications/P4.BC-lexical-normalization-and-payload-filters.md`
- `specifications/P4.D-repo-context-search-without-graph.md`
- `specifications/P4.E-chore-modular-decomposition-before-graph.md`
- `specifications/P5.BC-graph-seed-binding-and-expansion.md`
- `specifications/P5.D-rich-code-graph-edges.md`
- `specifications/P6.A-agent-reliability-freshness-and-recovery.md`

## Research inputs

Research artifacts and prompts that informed the accepted architecture:

- `research/rag-transfer/P4-rag-stack-transfer-from-rag-app.md`
- `research/hipporag/P5-hipporag-deep-research-prompt.md`
- `research/hipporag/P5-hipporag-research-analysis.md`
- `research/final-architecture/P6-final-architecture-artifact-index.md`
- `research/final-architecture/P6-final-architecture-gpt55-prompt.md`

These are inputs, not implementation contracts. If there is a conflict, prefer
`architecture/P6-final-architecture.md` and the active spec in
`specifications/`.

## Strategy

Model/backend strategy and long-lived direction:

- `strategy/P1.Y-EMBEDDING-PROFILES-AND-MODEL-STRATEGY.md`

## Dogfood

Practical validation runs:

- `dogfood/2026-05-07-private-backend-frontier-agent-checkpoint.md`

Reliability smoke entrypoint:

```powershell
py -3.12 scripts/agents/repo_semantic_reliability_smoke.py --repo C:\path\to\target-repo
```

The smoke is read-only by default and is the preferred quick check for HTTP
fallback, readiness, search probe, graph status, and no-rebuild lifecycle
discipline.

## Migration

Migration/history documents:

- `migration/REPO-SEMANTIC-SEARCH-MIGRATION-MAP.md`
