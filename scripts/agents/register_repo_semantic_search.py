"""Кроссплатформенная регистрация repo-semantic-search в Codex и Claude."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    return parser.parse_args()


def resolve_default_url() -> str:
    """Собрать default URL из live container env или fallback на 8011."""

    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
                "repo-semantic-mcp",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("SEMANTIC_MCP_HTTP_PORT="):
                port = line.split("=", 1)[1].strip()
                if port:
                    return f"http://127.0.0.1:{port}/mcp"
    except Exception:  # noqa: BLE001
        pass

    return "http://127.0.0.1:8011/mcp"


def backup_file(path: Path) -> None:
    """Сделать timestamp backup, если файл существует."""

    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path.with_name(f"{path.name}.{timestamp}.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def codex_command_config() -> tuple[str, list[str], int]:
    """Вернуть host-side wrapper команду для Codex."""

    script_dir = Path(__file__).resolve().parent
    if platform.system().lower().startswith("win"):
        return (
            "pwsh",
            [
                "-NoLogo",
                "-NoProfile",
                "-File",
                str(script_dir / "codex_repo_semantic_stdio.ps1"),
            ],
            900,
        )
    return (
        "bash",
        [str(script_dir / "codex_repo_semantic_stdio.sh")],
        900,
    )


def update_codex_config(path: Path) -> None:
    """Обновить конфиг Codex через stdio-wrapper секцию mcp_servers."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    backup_file(path)
    content = path.read_text(encoding="utf-8")
    section_pattern = re.compile(
        r"(?ms)^\[mcp_servers\.repo-semantic-search\]\n(?:.+\n)*?(?=^\[|\Z)"
    )
    command, args, startup_timeout_sec = codex_command_config()
    args_literal = ", ".join(json.dumps(arg) for arg in args)
    section_body = (
        "[mcp_servers.repo-semantic-search]\n"
        f'command = "{command}"\n'
        f"args = [{args_literal}]\n"
        f"startup_timeout_sec = {startup_timeout_sec}\n"
    )
    if section_pattern.search(content):
        updated = section_pattern.sub(section_body, content)
    else:
        separator = "" if content.endswith("\n") else "\n"
        updated = content + separator + "\n" + section_body
    path.write_text(updated, encoding="utf-8")


def update_claude_config(path: Path, url: str) -> None:
    """Обновить JSON-конфиг Claude."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    backup_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("mcpServers", {})
    data["mcpServers"]["repo-semantic-search"] = {"type": "http", "url": url}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Точка входа CLI."""

    args = parse_args()
    url = args.url or resolve_default_url()
    home = Path.home()
    update_codex_config(home / ".codex" / "config.toml")
    update_claude_config(home / ".claude.json", url)
    print(f"repo-semantic-search registered for Codex via stdio wrapper and for Claude via HTTP: {url}")


if __name__ == "__main__":
    main()
