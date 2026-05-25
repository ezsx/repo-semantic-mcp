param(
    [switch]$Build,
    [switch]$Clean,
    [switch]$ResetVolumes,
    [switch]$Gpu,
    [ValidateSet("cpu", "gpu", "gpu-qwen3", "gpu-bge-m3", "gpu-pplx-v100-infinity")]
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
        docker version *> $null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 3
    }

    throw "Docker daemon не стал доступен за ${TimeoutSec} секунд."
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

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not $Path -or -not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -Path $Path) {
        if ($line -match "^\s*$" -or $line -match "^\s*#") {
            continue
        }
        if ($line -match "^\s*$([regex]::Escape($Key))=(.*)$") {
            return $Matches[1].Trim().Trim("'`"")
        }
    }

    return $null
}

function Get-MergedEnvValue {
    param(
        $MergedEnv,
        [string]$Key
    )

    if (-not $MergedEnv) {
        return $null
    }

    $prop = $MergedEnv.PSObject.Properties[$Key]
    if ($prop) {
        return [string]$prop.Value
    }

    return $null
}

function Get-ContainerEnvValue {
    param(
        [string]$ContainerName,
        [string]$Key
    )

    try {
        $lines = docker inspect --format "{{range .Config.Env}}{{println .}}{{end}}" $ContainerName 2>$null
        foreach ($line in $lines) {
            if ($line -match "^\s*$([regex]::Escape($Key))=(.*)$") {
                return $Matches[1].Trim()
            }
        }
    }
    catch {
        return $null
    }

    return $null
}

function Invoke-BackendCatalogResolver {
    param(
        [string]$RepoRoot,
        [string]$Profile,
        [string]$EnvFilePath
    )

    $resolverScript = Join-Path $RepoRoot "scripts\\runtime\\repo_semantic_backend_catalog.py"
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue

    $args = @($resolverScript, "--profile", $Profile)
    if ($EnvFilePath) {
        $args += @("--env-file", $EnvFilePath)
    }

    if ($pythonCmd) {
        $output = & py -3.12 @args
    }
    else {
        $output = & python @args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось разрешить backend catalog для profile '$Profile'."
    }

    return $output | ConvertFrom-Json -Depth 20
}

function Get-SemanticMcpPort {
    param(
        [string]$EnvFilePath,
        $MergedEnv
    )

    $containerPort = Get-ContainerEnvValue -ContainerName "repo-semantic-mcp" -Key "SEMANTIC_MCP_HTTP_PORT"
    if ($containerPort) {
        return $containerPort
    }

    $envPort = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_HTTP_PORT"
    if (-not $envPort) {
        $envPort = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_HTTP_PORT"
    }
    if ($envPort) {
        return $envPort
    }

    return "8011"
}

function Ensure-ManagedExternalEmbedder {
    param(
        [string]$EnvFilePath,
        $MergedEnv
    )

    $enabled = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_EMBEDDER_ENABLED"
    if (-not $enabled) {
        $enabled = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_EMBEDDER_ENABLED"
    }
    if ($enabled -notin @("1", "true", "True", "yes", "on")) {
        return
    }

    $helperScript = Join-Path $PSScriptRoot "ensure_repo_semantic_gpu_server.ps1"
    $distro = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_DISTRO"
    if (-not $distro) {
        $distro = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_DISTRO"
    }
    if (-not $distro) {
        $distro = "Ubuntu-22.04"
    }
    $venvPath = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_VENV_PATH"
    if (-not $venvPath) {
        $venvPath = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_VENV_PATH"
    }
    if (-not $venvPath) {
        $venvPath = "/home/ezsx/infinity-env"
    }
    $modelPath = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_EMBEDDING_MODEL_PATH"
    if (-not $modelPath) {
        $modelPath = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_EMBEDDING_MODEL_PATH"
    }
    if (-not $modelPath) {
        $modelPath = "/mnt/c/llms/models/pplx-embed-v1-0.6B"
    }
    $serverPort = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_EMBEDDER_PORT"
    if (-not $serverPort) {
        $serverPort = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_EMBEDDER_PORT"
    }
    if (-not $serverPort) {
        $teiUrl = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_TEI_URL"
        if (-not $teiUrl) {
            $teiUrl = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_TEI_URL"
        }
        if ($teiUrl -match ":(\d+)(/|$)") {
            $serverPort = $Matches[1]
        }
        else {
            $serverPort = "8084"
        }
    }
    $cudaVisibleDevices = Get-MergedEnvValue -MergedEnv $MergedEnv -Key "SEMANTIC_MCP_WSL_CUDA_VISIBLE_DEVICES"
    if (-not $cudaVisibleDevices) {
        $cudaVisibleDevices = Get-EnvValue -Path $EnvFilePath -Key "SEMANTIC_MCP_WSL_CUDA_VISIBLE_DEVICES"
    }
    if (-not $cudaVisibleDevices) {
        $cudaVisibleDevices = "0"
    }

    & pwsh -NoLogo -NoProfile -File $helperScript `
        -Distro $distro `
        -VenvPath $venvPath `
        -ModelPath $modelPath `
        -CudaVisibleDevices $cudaVisibleDevices `
        -Port ([int]$serverPort)
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось поднять managed external embedder для repo-semantic-search."
    }
}

function Test-SemanticReadyRoute {
    param([string]$BaseUrl)

    try {
        $response = & curl.exe --silent --show-error --noproxy "*" --fail "$BaseUrl/readyz" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $response) {
            return $false
        }
        $payload = $response | ConvertFrom-Json
        return $payload.ready -eq $true
    }
    catch {
        return $false
    }
}

function Get-LogicalRepoRootForActivation {
    param(
        [string]$BaseUrl,
        [string]$ResolvedTargetRepo
    )

    if ($ResolvedTargetRepo) {
        return $ResolvedTargetRepo
    }

    try {
        $response = & curl.exe --silent --show-error --noproxy "*" --fail "$BaseUrl/statusz" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $response) {
            return $null
        }
        $payload = $response | ConvertFrom-Json
        if ($payload.index_status.repo_root) {
            return [string]$payload.index_status.repo_root
        }
        if ($payload.details.repo_root) {
            return [string]$payload.details.repo_root
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-RuntimeEmbeddingBackendId {
    param([string]$BaseUrl)

    try {
        $response = & curl.exe --silent --show-error --noproxy "*" --fail "$BaseUrl/statusz" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $response) {
            return $null
        }
        $payload = $response | ConvertFrom-Json
        if ($payload.index_status.embedding_backend_id) {
            return [string]$payload.index_status.embedding_backend_id
        }
    }
    catch {
        return $null
    }

    return $null
}

function Invoke-ActivateRuntimeRepo {
    param(
        [string]$RepoRoot,
        [string]$Port
    )

    if (-not $RepoRoot) {
        return
    }

    $script = @'
import anyio
import sys
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main() -> None:
    repo_root = sys.argv[1]
    port = sys.argv[2]
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("activate_repo", {"repo_root": repo_root})

anyio.run(main)
'@

    $script | docker exec -i repo-semantic-mcp python3 - $RepoRoot $Port | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось активировать runtime repo '$RepoRoot' после host-side switch."
    }
}

function Invoke-SelectRuntimeEmbeddingBackend {
    param(
        [string]$BackendId,
        [string]$Port
    )

    if (-not $BackendId) {
        return
    }

    $script = @'
import anyio
import sys
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main() -> None:
    backend_id = sys.argv[1]
    port = sys.argv[2]
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("set_role_backend", {"role": "embedding", "backend_id": backend_id})

anyio.run(main)
'@

    $script | docker exec -i repo-semantic-mcp python3 - $BackendId $Port | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось синхронизировать selected embedding backend '$BackendId' после explicit start."
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$deployDir = Join-Path $repoRoot "deploy\\repo-semantic-search"
$composeFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.yml"
$composeGpuFile = Join-Path $repoRoot "deploy\\repo-semantic-search\\docker-compose.repo-semantic-search.gpu.yml"

if (-not $Profile) {
    $Profile = "gpu"
}

$backendResolution = Invoke-BackendCatalogResolver -RepoRoot $repoRoot -Profile $Profile -EnvFilePath $EnvFile
$EnvFile = [string]$backendResolution.env_file
if (-not $EnvFile) {
    throw "Для profile '$Profile' не найден env file. Candidates: $($backendResolution.env_candidates -join ', ')"
}
$envLayers = @($backendResolution.env_layers)
$mergedEnv = $backendResolution.merged_env

$backend = $backendResolution.backend
$backendId = [string]$backend.backend_id
$launchMode = [string]$backend.config_blob.launch_mode
$useGpuCompose = [bool]$backend.config_blob.use_gpu_compose
$backendEndpoint = [string]$backend.endpoint
if (-not $launchMode) {
    throw "Backend '$backendId' не содержит launch_mode в catalog."
}

Wait-DockerReady -TimeoutSec 120

if ($TargetRepoPath) {
    $resolvedTargetRepo = (Resolve-Path $TargetRepoPath).Path
    $env:SEMANTIC_MCP_TARGET_REPO_PATH = $resolvedTargetRepo
    $env:SEMANTIC_MCP_REPO_ROOT = "/target_repo"
    $env:SEMANTIC_MCP_LOGICAL_REPO_ROOT = $resolvedTargetRepo
}

$composeArgs = @("compose", "-f", $composeFile)
if ($useGpuCompose) {
    $composeArgs += @("-f", $composeGpuFile)
}
$extraComposeFilesProp = $backend.config_blob.PSObject.Properties["compose_extra_files"]
if ($extraComposeFilesProp) {
    foreach ($extraComposeFile in @($extraComposeFilesProp.Value)) {
        if (-not $extraComposeFile) {
            continue
        }
        $extraComposePath = [string]$extraComposeFile
        if (-not [System.IO.Path]::IsPathRooted($extraComposePath)) {
            $extraComposePath = Join-Path $deployDir $extraComposePath
        }
        $composeArgs += @("-f", $extraComposePath)
    }
}
if ($envLayers.Count -gt 0) {
    foreach ($envLayer in $envLayers) {
        if ($envLayer) {
            $composeArgs += @("--env-file", [string]$envLayer)
        }
    }
}
elseif ($EnvFile) {
    $composeArgs += @("--env-file", $EnvFile)
}

if ($launchMode -eq "managed_host") {
    Ensure-ManagedExternalEmbedder -EnvFilePath $EnvFile -MergedEnv $mergedEnv
}

Write-Host "profile: $Profile"
if ($EnvFile) {
    Write-Host "env file: $EnvFile"
}
if ($envLayers.Count -gt 1) {
    Write-Host "env layers: $($envLayers -join ', ')"
}
Write-Host "backend id: $backendId"
Write-Host "launch mode: $launchMode"
if ($backendEndpoint) {
    Write-Host "backend endpoint: $backendEndpoint"
}
if ($TargetRepoPath) {
    Write-Host "target repo: $resolvedTargetRepo"
}
else {
    Write-Host "target repo: compose default"
}

if ($Clean) {
    $downArgs = $composeArgs + @("down", "--remove-orphans")
    if ($ResetVolumes) {
        $downArgs += "--volumes"
    }
    docker @downArgs | Out-Host
}

switch ($launchMode) {
    "bundled_compose" {
        $composeUpArgs = $composeArgs + @("up", "-d")
        if ($Build) {
            $composeUpArgs += "--build"
        }
        docker @composeUpArgs | Out-Host
    }
    "managed_host" {
        if ($Build) {
            docker @($composeArgs + @("build", "repo-semantic-mcp")) | Out-Host
        }
        docker @($composeArgs + @("up", "-d", "qdrant")) | Out-Host
        docker @($composeArgs + @("up", "-d", "--no-deps", "repo-semantic-mcp")) | Out-Host
    }
    "external_manual" {
        if ($Build) {
            docker @($composeArgs + @("build", "repo-semantic-mcp")) | Out-Host
        }
        docker @($composeArgs + @("up", "-d", "qdrant")) | Out-Host
        docker @($composeArgs + @("up", "-d", "--no-deps", "repo-semantic-mcp")) | Out-Host
    }
    default {
        throw "Неподдерживаемый launch_mode '$launchMode' для backend '$backendId'."
    }
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $status = docker ps --filter "name=^repo-semantic-mcp$" --format "{{.Status}}"
    if ($status -like "Up*") {
        $mcpPort = Get-SemanticMcpPort -EnvFilePath $EnvFile -MergedEnv $mergedEnv
        $baseUrl = "http://127.0.0.1:$mcpPort"
        if (Test-SemanticReadyRoute -BaseUrl $baseUrl) {
            $logicalRepoRoot = Get-LogicalRepoRootForActivation -BaseUrl $baseUrl -ResolvedTargetRepo $resolvedTargetRepo
            Invoke-ActivateRuntimeRepo -RepoRoot $logicalRepoRoot -Port $mcpPort
            $runtimeBackendId = Get-RuntimeEmbeddingBackendId -BaseUrl $baseUrl
            Invoke-SelectRuntimeEmbeddingBackend -BackendId $runtimeBackendId -Port $mcpPort
            Write-Host "repo-semantic-search ready: host-visible readiness is green on $baseUrl/readyz"
            Write-Host "mcp endpoint: $baseUrl/mcp"
            exit 0
        }
    }
    Start-Sleep -Seconds 5
}

throw "repo-semantic-search не стал готов за ${TimeoutSec} секунд."
