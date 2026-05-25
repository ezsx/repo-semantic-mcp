param(
    [string]$Profile = "gpu",
    [string]$RepoPath = "C:\\cursor_mcp\\repo-semantic-mcp",
    [string]$Query = "repo semantic backend registry",
    [switch]$Build,
    [switch]$Clean,
    [switch]$SkipEnsure
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ensureScript = Join-Path $scriptDir "ensure_repo_semantic_search.ps1"
$runtimeSmokeScript = Join-Path $scriptDir "..\\runtime\\repo_semantic_boot_smoke.py"
$resolvedRepoPath = (Resolve-Path $RepoPath).Path
$containerName = "repo-semantic-mcp"

if (-not $SkipEnsure) {
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
}

$httpPort = docker inspect --format "{{range .Config.Env}}{{println .}}{{end}}" $containerName 2>$null |
    Select-String "^SEMANTIC_MCP_HTTP_PORT=" |
    ForEach-Object { $_.Line.Split("=", 2)[1] } |
    Select-Object -First 1
if (-not $httpPort) {
    $httpPort = "8011"
}

$baseUrl = "http://127.0.0.1:$httpPort"

Get-Content $runtimeSmokeScript -Raw | docker exec -i $containerName python3 - --base-url $baseUrl --query $Query
