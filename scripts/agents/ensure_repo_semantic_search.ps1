param(
    [switch]$Build,
    [switch]$Clean,
    [switch]$Gpu,
    [ValidateSet("cpu", "gpu", "gpu-qwen3", "gpu-bge-m3")]
    [string]$Profile,
    [string]$EnvFile,
    [string]$TargetRepoPath,
    [int]$TimeoutSec = 1800
)

$ErrorActionPreference = "Stop"

function Wait-DockerReady {
    param([int]$TimeoutSec)

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            docker version | Out-Null
            return
        }
        catch {
            Start-Sleep -Seconds 3
        }
    }

    throw "Docker daemon не стал доступен за ${TimeoutSec} секунд."
}

function Test-SemanticMcpProtocol {
    $script = @'
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8011/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()
            print("OK")

anyio.run(main)
'@

    try {
        $script | docker exec -i repo-semantic-mcp python - 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-ExistingPath {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$deployDir = Join-Path $repoRoot "deploy\\repo-semantic-search"
$composeFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.yml"
$composeGpuFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.gpu.yml"

if (-not $Profile) {
    $Profile = if ($Gpu) { "gpu" } else { "cpu" }
}

$useGpuCompose = $Profile -ne "cpu" -or $Gpu

if (-not $EnvFile) {
    switch ($Profile) {
        "cpu" {
            $EnvFile = Resolve-ExistingPath @(
                (Join-Path $deployDir ".env"),
                (Join-Path $deployDir ".env.example")
            )
        }
        "gpu" {
            $EnvFile = Resolve-ExistingPath @(
                (Join-Path $deployDir ".env.gpu.qwen3"),
                (Join-Path $deployDir ".env.gpu.qwen3.example"),
                (Join-Path $deployDir ".env.gpu"),
                (Join-Path $deployDir ".env.gpu.example"),
                (Join-Path $deployDir ".env.gpu.bge-m3"),
                (Join-Path $deployDir ".env.gpu.bge-m3.example")
            )
        }
        "gpu-qwen3" {
            $EnvFile = Resolve-ExistingPath @(
                (Join-Path $deployDir ".env.gpu.qwen3"),
                (Join-Path $deployDir ".env.gpu"),
                (Join-Path $deployDir ".env.gpu.qwen3.example"),
                (Join-Path $deployDir ".env.gpu.example")
            )
        }
        "gpu-bge-m3" {
            $EnvFile = Resolve-ExistingPath @(
                (Join-Path $deployDir ".env.gpu.bge-m3"),
                (Join-Path $deployDir ".env.gpu.bge-m3.example")
            )
        }
    }
}

Wait-DockerReady -TimeoutSec 120

if ($TargetRepoPath) {
    $resolvedTargetRepo = (Resolve-Path $TargetRepoPath).Path
    $env:SEMANTIC_MCP_TARGET_REPO_PATH = $resolvedTargetRepo
    $env:SEMANTIC_MCP_REPO_ROOT = "/target_repo"
}

$composeArgs = @("compose", "-f", $composeFile)
if ($useGpuCompose) {
    $composeArgs += @("-f", $composeGpuFile)
}
if ($EnvFile) {
    $composeArgs += @("--env-file", $EnvFile)
}

Write-Host "profile: $Profile"
if ($EnvFile) {
    Write-Host "env file: $EnvFile"
}
if ($TargetRepoPath) {
    Write-Host "target repo: $resolvedTargetRepo"
}
else {
    Write-Host "target repo: compose default"
}

if ($Clean) {
    docker @($composeArgs + @("down", "--remove-orphans")) | Out-Host
}

$composeArgs += @("up", "-d")
if ($Build) {
    $composeArgs += "--build"
}
docker @composeArgs | Out-Host

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $status = docker ps --filter "name=^repo-semantic-mcp$" --format "{{.Status}}"
    if ($status -like "Up*") {
        if (Test-SemanticMcpProtocol) {
            Write-Host "repo-semantic-search ready: MCP protocol is responding on http://127.0.0.1:8011/mcp"
            exit 0
        }
    }
    Start-Sleep -Seconds 5
}

throw "repo-semantic-search не стал готов за ${TimeoutSec} секунд."
