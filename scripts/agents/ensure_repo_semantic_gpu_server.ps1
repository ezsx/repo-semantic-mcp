param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$VenvPath = "/home/ezsx/infinity-env",
    [string]$ModelPath = "/mnt/c/llms/models/pplx-embed-v1-0.6B",
    [string]$CudaVisibleDevices = "0",
    [int]$Port = 8084,
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"

function Test-GpuServerHealth {
    param([int]$ServerPort)

    try {
        $response = & curl.exe --silent --show-error --noproxy "*" --fail "http://127.0.0.1:$ServerPort/health" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $response) {
            return $false
        }
        $payload = $response | ConvertFrom-Json
        return $payload.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Quote-BashLiteral {
    param([string]$Value)

    if ($null -eq $Value) {
        return "''"
    }

    $replacement = "'""'""'"
    return "'" + $Value.Replace("'", $replacement) + "'"
}

function Convert-WindowsPathToWsl {
    param([string]$WindowsPath)

    $resolved = (Resolve-Path $WindowsPath).Path
    if ($resolved -notmatch '^(?<drive>[A-Za-z]):(?<rest>.*)$') {
        throw "Не удалось распознать Windows path для WSL conversion: $resolved"
    }

    $drive = $Matches["drive"].ToLowerInvariant()
    $rest = $Matches["rest"].Replace("\", "/")
    return "/mnt/$drive$rest"
}

if (Test-GpuServerHealth -ServerPort $Port) {
    Write-Host "repo-semantic GPU server is already healthy on http://127.0.0.1:$Port/health"
    exit 0
}

$serverScriptWindowsPath = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path "scripts\\runtime\\repo_semantic_gpu_server.py"
$serverScriptWslPath = Convert-WindowsPathToWsl -WindowsPath $serverScriptWindowsPath
$launcherWindowsPath = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path "scripts\\runtime\\start_repo_semantic_gpu_server.sh"
$launcherWslPath = Convert-WindowsPathToWsl -WindowsPath $launcherWindowsPath

$windowsLogDir = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path "artifacts"
New-Item -ItemType Directory -Force -Path $windowsLogDir | Out-Null
$stdoutLog = Join-Path $windowsLogDir "repo-semantic-gpu-server-$Port.stdout.log"
$stderrLog = Join-Path $windowsLogDir "repo-semantic-gpu-server-$Port.stderr.log"
Remove-Item -Force $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

$killCommand = "pkill -f repo_semantic_gpu_server.py >/dev/null 2>&1 || true"
& wsl.exe -d $Distro bash -lc $killCommand | Out-Null
$null = $LASTEXITCODE

$launcherArgs = @(
    "-d",
    $Distro,
    "bash",
    $launcherWslPath,
    $serverScriptWslPath,
    "$VenvPath/bin/activate",
    $ModelPath,
    $Port.ToString(),
    $CudaVisibleDevices
)

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList $launcherArgs `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

if (-not $process) {
    throw "Не удалось создать процесс для WSL GPU server."
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    if (Test-GpuServerHealth -ServerPort $Port) {
        Write-Host "repo-semantic GPU server is healthy on http://127.0.0.1:$Port/health"
        exit 0
    }
    if ($process.HasExited) {
        break
    }
    Start-Sleep -Seconds 2
}

$stderrTail = if (Test-Path $stderrLog) { Get-Content -Path $stderrLog -Tail 80 } else { @() }
$stdoutTail = if (Test-Path $stdoutLog) { Get-Content -Path $stdoutLog -Tail 80 } else { @() }
if ($stdoutTail) {
    Write-Host "--- stdout ---"
    $stdoutTail | Out-Host
}
if ($stderrTail) {
    Write-Host "--- stderr ---"
    $stderrTail | Out-Host
}
throw "repo-semantic GPU server не стал готов за ${TimeoutSec} секунд."
