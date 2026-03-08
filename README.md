# Repo Semantic MCP

`repo-semantic-mcp` is a standalone semantic search MCP for code and docs repositories.

## What it provides

- `Qdrant` as persistent vector store
- `TEI` or `FastEmbed` as embedding backend
- separate `code` and `docs` collections
- semantic and hybrid retrieval tools for MCP clients
- full rebuild, partial reindex and watch mode

## Current runtime profile

- default shared deployment: `tei_http`
- default model: `intfloat/multilingual-e5-small`
- default transport: HTTP MCP on `127.0.0.1:8011/mcp`

## Quick start

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Build
```

This starts:
- `repo-semantic-qdrant`
- `repo-semantic-tei`
- `repo-semantic-mcp`

## Layout

- `apps/repo-semantic-mcp/` - MCP app entrypoint and image build inputs
- `services/repo_semantic/` - indexer, chunkers, embeddings, Qdrant integration
- `deploy/repo-semantic-search/` - compose stack and env contract
- `scripts/agents/` - helper scripts for startup and MCP registration
- `docs/` - specification and migration notes

## Next cleanup

- vendor-neutral docs and defaults
- dedicated GPU deploy profile
- packaging and GitHub publish
