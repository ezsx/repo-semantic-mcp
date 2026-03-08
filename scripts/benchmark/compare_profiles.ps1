param(
    [string]$RepoPath = ".",
    [ValidateSet("cpu", "gpu-bge-m3", "gpu", "gpu-qwen3")]
    [string]$BaseProfile = "cpu",
    [ValidateSet("cpu", "gpu-bge-m3", "gpu", "gpu-qwen3")]
    [string]$CandidateProfile = "gpu",
    [string]$QueriesFile = "scripts/benchmark/queries.repo-semantic.json",
    [ValidateSet("semantic_search", "hybrid_search")]
    [string]$Tool = "semantic_search",
    [int]$WaitForIndexSec = 300
)

$ErrorActionPreference = "Stop"

function Invoke-ProfileBenchmark {
    param(
        [string]$Label,
        [string]$Profile
    )

    Write-Host "=== $Label / $Profile ==="
    pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Build -Clean -Profile $Profile -TargetRepoPath $RepoPath | Out-Host

    $tmpFile = [System.IO.Path]::GetTempFileName()
    try {
        Write-Host "running benchmark, wait_for_index_sec=$WaitForIndexSec"
        docker exec repo-semantic-mcp python /repo/scripts/benchmark/run_semantic_benchmark.py --label $Label --queries-file /repo/$QueriesFile --tool $Tool --wait-for-index-sec $WaitForIndexSec 2>&1 |
            Tee-Object -FilePath $tmpFile | Out-Host
        return Get-Content -Path $tmpFile -Raw
    }
    finally {
        Remove-Item $tmpFile -ErrorAction SilentlyContinue
    }
}

$baseResult = Invoke-ProfileBenchmark -Label "baseline" -Profile $BaseProfile
$candidateResult = Invoke-ProfileBenchmark -Label "candidate" -Profile $CandidateProfile

Write-Host "BASELINE"
Write-Host "profile=$BaseProfile"
Write-Host $baseResult
Write-Host ""
Write-Host "CANDIDATE"
Write-Host "profile=$CandidateProfile"
Write-Host $candidateResult
