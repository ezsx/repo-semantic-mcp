param(
    [switch]$Build,
    [switch]$Gpu,
    [string]$EnvFile,
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$deployDir = Join-Path $repoRoot "deploy\\repo-semantic-search"
$composeFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.yml"
$composeGpuFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.gpu.yml"

if (-not $EnvFile) {
    if ($Gpu) {
        $gpuEnv = Join-Path $deployDir ".env.gpu"
        if (Test-Path $gpuEnv) {
            $EnvFile = $gpuEnv
        }
    }
    if (-not $EnvFile) {
        $defaultEnv = Join-Path $deployDir ".env"
        if (Test-Path $defaultEnv) {
            $EnvFile = $defaultEnv
        }
    }
}

Wait-DockerReady -TimeoutSec 120

$composeArgs = @("compose", "-f", $composeFile)
if ($Gpu) {
    $composeArgs += @("-f", $composeGpuFile)
}
if ($EnvFile) {
    $composeArgs += @("--env-file", $EnvFile)
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
