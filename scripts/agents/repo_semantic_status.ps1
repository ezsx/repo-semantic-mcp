param(
    [string]$ContainerName = "repo-semantic-mcp",
    [string]$BaseUrl
)

$ErrorActionPreference = "Stop"

$status = docker ps --filter "name=^$ContainerName$" --format "{{.Status}}"
if (-not $status) {
    throw "Container '$ContainerName' is not running."
}

if (-not $BaseUrl) {
    $httpPort = docker inspect --format "{{range .Config.Env}}{{println .}}{{end}}" $ContainerName 2>$null |
        Select-String "^SEMANTIC_MCP_HTTP_PORT=" |
        ForEach-Object { $_.Line.Split("=", 2)[1] } |
        Select-Object -First 1
    if (-not $httpPort) {
        $httpPort = "8011"
    }
    $BaseUrl = "http://127.0.0.1:$httpPort"
}

$response = & curl.exe --silent --show-error --noproxy "*" --fail "$BaseUrl/statusz"
($response | ConvertFrom-Json) | ConvertTo-Json -Depth 10
