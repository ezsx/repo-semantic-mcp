param(
    [string]$Url = "http://127.0.0.1:8011/mcp"
)

$ErrorActionPreference = "Stop"

function Backup-File {
    param([string]$Path)

    if (Test-Path $Path) {
        $timestamp = Get-Date -Format "yyyyMMddHHmmss"
        Copy-Item $Path "$Path.$timestamp.bak"
    }
}

function Update-CodexConfig {
    param([string]$Path, [string]$Url)

    if (!(Test-Path $Path)) {
        Set-Content -Path $Path -Value "model = `"gpt-5.4`"`r`npersonality = `"pragmatic`"`r`n"
    }

    Backup-File -Path $Path

    $content = Get-Content -Path $Path -Raw
    $sectionPattern = '(?ms)^\[mcp_servers\.repo-semantic-search\]\r?\n(?:.+\r?\n)*?(?=^\[|\z)'
    $sectionBody = "[mcp_servers.repo-semantic-search]`r`nurl = `"$Url`"`r`n"

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
        Set-Content -Path $Path -Value "{}"
    }

    Backup-File -Path $Path

    $env:CLAUDE_CONFIG_PATH = $Path
    $env:SEMANTIC_MCP_URL = $Url
    @'
import json
import os
from pathlib import Path

path = Path(os.environ["CLAUDE_CONFIG_PATH"])
url = os.environ["SEMANTIC_MCP_URL"]

if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
else:
    data = {}

data.setdefault("mcpServers", {})
data["mcpServers"]["repo-semantic-search"] = {"type": "http", "url": url}
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
'@ | py -3.12 -
}

$codexPath = Join-Path $env:USERPROFILE ".codex\config.toml"
$claudePath = Join-Path $env:USERPROFILE ".claude.json"

Update-CodexConfig -Path $codexPath -Url $Url
Update-ClaudeConfig -Path $claudePath -Url $Url

Write-Host "repo-semantic-search registered in Codex and Claude."
