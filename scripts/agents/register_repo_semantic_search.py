"""Кроссплатформенная регистрация repo-semantic-search в Codex и Claude."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011/mcp")
    return parser.parse_args()


def backup_file(path: Path) -> None:
    """Сделать timestamp backup, если файл существует."""

    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path.with_name(f"{path.name}.{timestamp}.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def update_codex_config(path: Path, url: str) -> None:
    """Обновить конфиг Codex через секцию mcp_servers."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('model = "gpt-5.4"\npersonality = "pragmatic"\n', encoding="utf-8")

    backup_file(path)
    content = path.read_text(encoding="utf-8")
    section_pattern = re.compile(
        r"(?ms)^\[mcp_servers\.repo-semantic-search\]\n(?:.+\n)*?(?=^\[|\Z)"
    )
    section_body = f'[mcp_servers.repo-semantic-search]\nurl = "{url}"\n'
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
    home = Path.home()
    update_codex_config(home / ".codex" / "config.toml", args.url)
    update_claude_config(home / ".claude.json", args.url)
    print("repo-semantic-search registered in Codex and Claude.")


if __name__ == "__main__":
    main()
