param(
    [string]$Url
)

$ErrorActionPreference = "Stop"

function Backup-File {
    param([string]$Path)

    if (Test-Path $Path) {
        $timestamp = Get-Date -Format "yyyyMMddHHmmss"
        Copy-Item $Path "$Path.$timestamp.bak"
    }
}

function Get-DefaultUrl {
    $httpPort = docker inspect --format "{{range .Config.Env}}{{println .}}{{end}}" repo-semantic-mcp 2>$null |
        Select-String "^SEMANTIC_MCP_HTTP_PORT=" |
        ForEach-Object { $_.Line.Split("=", 2)[1] } |
        Select-Object -First 1

    if (-not $httpPort) {
        $httpPort = "8011"
    }

    return "http://127.0.0.1:$httpPort/mcp"
}

function Get-CodexCommandConfig {
    $scriptPath = Join-Path $PSScriptRoot "codex_repo_semantic_stdio.ps1"
    return @{
        command = "pwsh"
        args = @(
            "-NoLogo",
            "-NoProfile",
            "-File",
            $scriptPath
        )
        startup_timeout_sec = 900
    }
}

function ConvertTo-TomlStringLiteral {
    param([string]$Value)

    $escaped = $Value.Replace('\', '\\').Replace('"', '\"')
    return "`"$escaped`""
}

function Update-CodexConfig {
    param([string]$Path)

    if (!(Test-Path $Path)) {
        $parent = Split-Path -Parent $Path
        if ($parent) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Set-Content -Path $Path -Value ""
    }

    Backup-File -Path $Path

    $content = Get-Content -Path $Path -Raw
    $sectionPattern = '(?ms)^\[mcp_servers\.repo-semantic-search\]\r?\n(?:.+\r?\n)*?(?=^\[|\z)'
    $config = Get-CodexCommandConfig
    $argsLiteral = ($config.args | ForEach-Object { ConvertTo-TomlStringLiteral $_ }) -join ", "
    $sectionBody = @"
[mcp_servers.repo-semantic-search]
command = $(ConvertTo-TomlStringLiteral $config.command)
args = [$argsLiteral]
startup_timeout_sec = $($config.startup_timeout_sec)
"@
    $sectionBody = $sectionBody.Replace("`n", "`r`n")

    if ($content -match $sectionPattern) {
        $updated = [regex]::Replace($content, $sectionPattern, $sectionBody)
    }
    else {
        $separator = if ($content.EndsWith("`n")) { "" } else { "`r`n" }
        $updated = $content + $separator + "`r`n" + $sectionBody
    }

    Set-Content -Path $Path -Value $updated
}

function Update-ClaudeConfig {
    param([string]$Path, [string]$Url)

    if (!(Test-Path $Path)) {
        $parent = Split-Path -Parent $Path
        if ($parent) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Set-Content -Path $Path -Value "{}"
    }

    Backup-File -Path $Path

    $raw = Get-Content -Path $Path -Raw
    $data = if ($raw.Trim()) {
        $raw | ConvertFrom-Json -AsHashtable
    }
    else {
        @{}
    }

    if (-not $data) {
        $data = @{}
    }
    if (-not $data.ContainsKey("mcpServers")) {
        $data["mcpServers"] = @{}
    }

    $data["mcpServers"]["repo-semantic-search"] = @{
        type = "http"
        url = $Url
    }

    $json = $data | ConvertTo-Json -Depth 10
    Set-Content -Path $Path -Value $json
}

$codexPath = Join-Path $env:USERPROFILE ".codex\config.toml"
$claudePath = Join-Path $env:USERPROFILE ".claude.json"

if (-not $Url) {
    $Url = Get-DefaultUrl
}

Update-CodexConfig -Path $codexPath
Update-ClaudeConfig -Path $claudePath -Url $Url

Write-Host "repo-semantic-search registered for Codex via stdio wrapper and for Claude via HTTP: $Url"
