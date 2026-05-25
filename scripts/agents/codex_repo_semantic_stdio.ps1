param(
    [string]$ContainerName = "repo-semantic-mcp",
    [int]$ContainerStartTimeoutSec = 120,
    [int]$ReadyTimeoutSec = 900
)

$ErrorActionPreference = "Stop"
$script:WrapperScriptDir = $PSScriptRoot

function Fail {
    param([string]$Message)

    [Console]::Error.WriteLine($Message)
    exit 1
}

function Test-ContainerExists {
    param([string]$Name)

    docker inspect $Name *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-ContainerRunning {
    param(
        [string]$Name,
        [int]$TimeoutSec
    )

    if (-not (Test-ContainerExists -Name $Name)) {
        return $false
    }

    $running = (docker inspect --format "{{.State.Running}}" $Name 2>$null).Trim()
    if ($running -eq "true") {
        return $true
    }

    docker start $Name *> $null

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $running = (docker inspect --format "{{.State.Running}}" $Name 2>$null).Trim()
        if ($running -eq "true") {
            return $true
        }
        Start-Sleep -Seconds 2
    }

    return $false
}

function Get-ContainerEnvValue {
    param(
        [string]$Name,
        [string]$Key
    )

    try {
        $lines = docker inspect --format "{{range .Config.Env}}{{println .}}{{end}}" $Name 2>$null
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

function Wait-Ready {
    param(
        [string]$BaseUrl,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = & curl.exe --silent --show-error --noproxy "*" --fail "$BaseUrl/readyz" 2>$null
            if ($LASTEXITCODE -eq 0 -and $response) {
                $payload = $response | ConvertFrom-Json
                if ($payload.ready -eq $true) {
                    return
                }
            }
        }
        catch {
            # readiness ещё не зелёный; продолжаем ждать
        }

        Start-Sleep -Seconds 2
    }

    Fail "repo-semantic-search container did not become ready within ${TimeoutSec} seconds."
}

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $script:WrapperScriptDir "..\..")).Path
}

function Get-PythonLauncher {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.12") },
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }
        & $candidate.Command @($candidate.Args + @("-c", "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)")) *> $null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }

    return $null
}

foreach ($dependency in @("repo-semantic-qdrant", "repo-semantic-tei")) {
    Ensure-ContainerRunning -Name $dependency -TimeoutSec 30 | Out-Null
}

if (-not (Ensure-ContainerRunning -Name $ContainerName -TimeoutSec $ContainerStartTimeoutSec)) {
    Fail "repo-semantic-search container '$ContainerName' is missing or did not start. Start it explicitly before using the Codex wrapper."
}

$httpPort = Get-ContainerEnvValue -Name $ContainerName -Key "SEMANTIC_MCP_HTTP_PORT"
if (-not $httpPort) {
    $httpPort = "8011"
}

$baseUrl = "http://127.0.0.1:$httpPort"
Wait-Ready -BaseUrl $baseUrl -TimeoutSec $ReadyTimeoutSec

$repoRoot = Get-RepoRoot
$hostProxyScript = Join-Path $repoRoot "scripts\runtime\repo_semantic_host_stdio_proxy.py"
$pythonLauncher = Get-PythonLauncher
if ($pythonLauncher -and (Test-Path $hostProxyScript)) {
    $env:SEMANTIC_MCP_PROXY_URL = "$baseUrl/mcp"
    & $pythonLauncher.Command @($pythonLauncher.Args + @($hostProxyScript))
    exit $LASTEXITCODE
}

& docker exec `
    -i `
    -e SEMANTIC_MCP_PROXY_URL=$baseUrl/mcp `
    $ContainerName `
    python3 /repo/scripts/runtime/repo_semantic_stdio_proxy.py

exit $LASTEXITCODE
