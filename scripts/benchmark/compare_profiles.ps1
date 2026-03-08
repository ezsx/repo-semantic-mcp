param(
    [string]$RepoPath = ".",
    [string]$BaseEnvFile = "deploy/repo-semantic-search/.env.gpu.example",
    [string]$CandidateEnvFile = "deploy/repo-semantic-search/.env.gpu.qwen3.example",
    [string]$QueriesFile = "scripts/benchmark/queries.repo-semantic.json",
    [ValidateSet("semantic_search", "hybrid_search")]
    [string]$Tool = "semantic_search"
)

$ErrorActionPreference = "Stop"

function Invoke-ProfileBenchmark {
    param(
        [string]$Label,
        [string]$EnvFile
    )

    pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Build -Clean -Gpu -EnvFile $EnvFile -TargetRepoPath $RepoPath | Out-Host
    docker exec repo-semantic-mcp python /repo/scripts/benchmark/run_semantic_benchmark.py --label $Label --queries-file /repo/$QueriesFile --tool $Tool
}

$baseResult = Invoke-ProfileBenchmark -Label "baseline" -EnvFile $BaseEnvFile
$candidateResult = Invoke-ProfileBenchmark -Label "candidate" -EnvFile $CandidateEnvFile

Write-Host "BASELINE"
Write-Host $baseResult
Write-Host ""
Write-Host "CANDIDATE"
Write-Host $candidateResult
