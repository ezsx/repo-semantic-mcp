param(
    [Parameter(Mandatory = $true)]
    [string]$RepoPath,
    [ValidateSet("gpu", "cpu", "gpu-bge-m3")]
    [string]$Profile = "gpu",
    [switch]$Clean,
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ensureScript = Join-Path $scriptDir "ensure_repo_semantic_search.ps1"
$statusScript = Join-Path $scriptDir "repo_semantic_status.ps1"
$resolvedRepoPath = (Resolve-Path $RepoPath).Path

if (-not $Build) {
    docker image inspect repo-semantic-search-repo-semantic-mcp *> $null
    if ($LASTEXITCODE -ne 0) {
        $Build = $true
    }
}

$ensureArgs = @(
    "-File", $ensureScript,
    "-Profile", $Profile,
    "-TargetRepoPath", $resolvedRepoPath
)
if ($Build) {
    $ensureArgs += "-Build"
}
if ($Clean) {
    $ensureArgs += "-Clean"
}

pwsh @ensureArgs

Write-Host ""
Write-Host "repo-semantic-search status:"
pwsh -File $statusScript
