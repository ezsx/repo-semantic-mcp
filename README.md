# repo-semantic-mcp

Standalone semantic MCP for repository search with:
- `Qdrant` as persistent vector store
- `TEI` or `FastEmbed` as embedding backend
- separate `code` and `docs` collections
- semantic and hybrid retrieval tools for MCP clients
- full rebuild, partial reindex, and watch mode

The runtime is container-first and intended to be reusable across repositories. The MCP server itself stays stable; you point it at the repository you want to index.

## Platform Support

- Windows: supported, including the main documented path
- Ubuntu/Linux: supported
- macOS: supported for CPU profile

GPU profile expectations:
- Windows: supported when Docker GPU passthrough works
- Linux: supported with NVIDIA Container Toolkit
- macOS: CPU only; no CUDA TEI path

## What It Solves

- keeps `code` and `docs` retrieval separate, so agents can search the right corpus intentionally
- works as a shared HTTP MCP for Codex and Claude
- supports CPU and GPU embedding profiles without mixing incompatible collections
- can be repointed to any local repository through `TargetRepoPath`

## Runtime Model

One running stack indexes one target repository at a time.

That means:
- open your target repository and start the MCP against its local path
- later switch to another project by restarting the same stack with a different `TargetRepoPath`

The MCP does not auto-detect the IDE folder. The target repository is an explicit launch parameter. This is intentional because it is deterministic and works the same for Codex and Claude.

## Profiles

### CPU default

- profile: `cpu_e5`
- backend: `tei_http`
- model: `intfloat/multilingual-e5-small`
- query format: `query: {query}`
- document format: `passage: {text}`

Use this when:
- you want the safest default
- the machine has no NVIDIA GPU
- colleagues need a low-friction setup

### GPU primary

- profile: `gpu_qwen3`
- backend: `tei_http`
- model: `Qwen/Qwen3-Embedding-0.6B`

Use this when:
- you have a working NVIDIA Docker stack
- you want the best current retrieval quality in this repo

### CPU fallback

If `Qwen3` is unhealthy or Docker GPU runtime is broken, the recommended fallback is not another heavy GPU model. The recommended fallback is:

- profile: `cpu_e5`
- backend: `tei_http`
- model: `intfloat/multilingual-e5-small`

This is slower but operationally predictable.

## Quick Start

### First setup on a new machine

1. Register the MCP once:

Windows:

```powershell
pwsh -File scripts/agents/register_repo_semantic_search.ps1
```

Linux/macOS:

```bash
python3 scripts/agents/register_repo_semantic_search.py
```

2. Build and start the GPU profile for your target repository:

Windows:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

Linux:

```bash
bash scripts/agents/ensure_repo_semantic_search.sh \
  --build \
  --profile gpu \
  --target-repo-path /path/to/target-repo
```

macOS:

```bash
bash scripts/agents/ensure_repo_semantic_search.sh \
  --build \
  --profile cpu \
  --target-repo-path /path/to/target-repo
```

3. Restart Codex or Claude if they were already open before MCP registration.

### CPU

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile cpu `
  -TargetRepoPath C:\path\to\target-repo
```

### GPU primary

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

### Clean restart against another repository

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Clean `
  -Profile gpu `
  -TargetRepoPath C:\some\other\repo
```

## Daily Flow For A Target Repository

If you work on the same target repository every day, the practical flow is:

1. start Docker Desktop
2. if the stack was already configured for that repo path, wait for it to come back
3. if you want an explicit readiness check, run:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

4. open that repository in Codex or Claude and use the same registered MCP endpoint

### What happens automatically

- containers restart with `restart: unless-stopped`
- the same MCP endpoint stays registered in Codex and Claude
- the stack keeps the same target repo and same index collections for that repo/profile

### What still requires an explicit action

- changing to another target repository
- changing profile
- forcing a clean restart

For those cases, rerun `ensure_repo_semantic_search.ps1` with the desired `-TargetRepoPath` or `-Profile`.

## Startup Time On This Machine

Measured on the current workstation with:
- `RTX 5060 Ti 16 GB`
- `Qwen/Qwen3-Embedding-0.6B`
- target repo: a medium-sized local repository

Observed end-to-end time from clean restart to MCP ready:
- first clean start after switching to `120-1.9`: about `2.5 min`
- repeated clean restart on the same repo/profile: about `1.5 min`

This profile now assumes:
- TEI image: `ghcr.io/huggingface/text-embeddings-inference:120-1.9`
- warmup budget: `SEMANTIC_MCP_TEI_MAX_BATCH_TOKENS=4096`

Operationally, treat `Qwen3` startup as roughly `1.5-2.5 minutes` on this machine.

Once the stack is ready, retrieval latency is much lower than startup latency. Startup is the expensive phase.

## Switching Between Projects

If you move from one project to another:

1. restart the stack with a new `-TargetRepoPath`
2. wait until `ensure_repo_semantic_search.ps1` reports MCP ready
3. keep using the same MCP endpoint in Codex or Claude

The clients do not need a different URL. Only the indexed target repository changes.

## Codex and Claude

Register the shared HTTP MCP:

```powershell
pwsh -File scripts/agents/register_repo_semantic_search.ps1
```

This updates:
- `%USERPROFILE%\.codex\config.toml`
- `%USERPROFILE%\.claude.json`

After registration:
- Codex and Claude talk to the same MCP endpoint
- the endpoint serves whichever repository was last started via `TargetRepoPath`

For Claude, this is practical because the integration is URL-based. You only need to restart the MCP stack when changing target repos, not reconfigure Claude each time.

Both Codex and Claude use the same registered URL-based MCP endpoint on all supported platforms.

## Concurrency

The MCP is a shared HTTP service. Codex and Claude can both hit the same endpoint.

Practical implication:
- once startup is complete, normal retrieval requests are fast enough for concurrent use
- if two agents query at the same time, they wait on the same running service rather than starting a second indexer
- the expensive phase is startup or full rebuild, not ordinary search

## Benchmarking

Compare GPU profiles on the same repository:

```powershell
pwsh -File scripts/benchmark/compare_profiles.ps1 `
  -RepoPath C:\cursor_mcp\repo-semantic-mcp `
  -BaseProfile cpu `
  -CandidateProfile gpu
```

Run a single benchmark pass:

```powershell
docker exec repo-semantic-mcp python /repo/scripts/benchmark/run_semantic_benchmark.py `
  --queries-file /repo/scripts/benchmark/queries.repo-semantic.json `
  --wait-for-index-sec 300
```

## Current Recommendation

- default CPU profile for broad compatibility: `intfloat/multilingual-e5-small`
- primary GPU profile: `Qwen/Qwen3-Embedding-0.6B`
- official fallback profile: `intfloat/multilingual-e5-small` on CPU

`bge-m3` remains available only as an experimental/debug profile. It is not the recommended fallback because its cold start is operationally too expensive.

## License

This repository is prepared for publication under `PolyForm Noncommercial 1.0.0`.

That means:
- copying and modification are allowed under the license terms
- commercial use is not allowed
- license notices must be preserved

This is source-available, not OSI open source. If later you want commercial use to be allowed, switch to another license explicitly.

## Repository Layout

- `apps/repo-semantic-mcp/` - MCP entrypoint and image build inputs
- `services/repo_semantic/` - indexer, chunkers, embeddings, search, Qdrant integration
- `deploy/repo-semantic-search/` - compose stack and env contract
- `scripts/agents/` - startup and registration helpers for PowerShell, Bash, and Python registration
- `scripts/benchmark/` - profile comparison and retrieval benchmark tools
- `docs/` - specifications and migration notes

## Operational Notes

- containers use `restart: unless-stopped`
- `Qdrant` data lives on a persistent Docker volume
- `TEI` model cache also lives on a persistent Docker volume
- one stack equals one active target repository
- collection names include a repo-specific key, so two different repositories do not silently share one index
- profile and model are part of the collection name to avoid index corruption across embeddings or query formats

## Common Scenarios

### I rebooted the PC and want to work on my target repository

Windows:

```powershell
pwsh -File C:\cursor_mcp\repo-semantic-mcp\scripts\agents\ensure_repo_semantic_search.ps1 `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

Linux:

```bash
bash /path/to/repo-semantic-mcp/scripts/agents/ensure_repo_semantic_search.sh \
  --profile gpu \
  --target-repo-path /path/to/target-repo
```

macOS:

```bash
bash /path/to/repo-semantic-mcp/scripts/agents/ensure_repo_semantic_search.sh \
  --profile cpu \
  --target-repo-path /path/to/target-repo
```

Wait until the script prints that MCP is ready. After that, open or continue your Codex or Claude session on that repository.

### I want to switch from one local repository to another

```powershell
pwsh -File C:\cursor_mcp\repo-semantic-mcp\scripts\agents\ensure_repo_semantic_search.ps1 `
  -Clean `
  -Profile gpu `
  -TargetRepoPath C:\path\to\other\repo
```

### Qwen3 is unhealthy or Docker GPU runtime is broken

```powershell
pwsh -File C:\cursor_mcp\repo-semantic-mcp\scripts\agents\ensure_repo_semantic_search.ps1 `
  -Clean `
  -Profile cpu `
  -TargetRepoPath C:\path\to\target-repo
```

### I changed the MCP code itself and need a rebuild

```powershell
pwsh -File C:\cursor_mcp\repo-semantic-mcp\scripts\agents\ensure_repo_semantic_search.ps1 `
  -Build `
  -Clean `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

## Next Publication Tasks

- optional: add a small release/roadmap section if you want public versioning from day one
