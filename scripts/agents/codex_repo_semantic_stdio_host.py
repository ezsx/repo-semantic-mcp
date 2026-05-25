"""Host-side launcher для Codex stdio MCP без shell buffering."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


CONTAINER_NAME = os.getenv("CONTAINER_NAME", "repo-semantic-mcp")
CONTAINER_START_TIMEOUT_SEC = int(os.getenv("CONTAINER_START_TIMEOUT_SEC", "120"))
READY_TIMEOUT_SEC = int(os.getenv("READY_TIMEOUT_SEC", "900"))


def fail(message: str) -> None:
    """Завершить launcher с понятной ошибкой."""

    print(message, file=sys.stderr)
    raise SystemExit(1)


def docker_output(*args: str) -> str:
    """Выполнить docker-команду и вернуть stdout."""

    result = subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def container_exists(name: str) -> bool:
    """Проверить существование контейнера."""

    return bool(docker_output("inspect", name))


def ensure_container_running(name: str, timeout_sec: int) -> bool:
    """Убедиться, что контейнер запущен."""

    if not container_exists(name):
        return False
    if docker_output("inspect", "--format", "{{.State.Running}}", name) == "true":
        return True
    subprocess.run(["docker", "start", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if docker_output("inspect", "--format", "{{.State.Running}}", name) == "true":
            return True
        time.sleep(2)
    return False


def get_container_env(name: str, key: str) -> str | None:
    """Прочитать env var из контейнера."""

    output = docker_output("inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", name)
    for line in output.splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def wait_ready(base_url: str, timeout_sec: int) -> None:
    """Дождаться зелёного `/readyz` у HTTP runtime."""

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/readyz", timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(2)
    fail(f"repo-semantic-search container did not become ready within {timeout_sec} seconds.")


def main() -> None:
    """Выполнить precheck и заменить процесс на docker exec stdio proxy."""

    for dependency in ("repo-semantic-qdrant", "repo-semantic-tei"):
        ensure_container_running(dependency, 30)

    if not ensure_container_running(CONTAINER_NAME, CONTAINER_START_TIMEOUT_SEC):
        fail(
            f"repo-semantic-search container '{CONTAINER_NAME}' is missing or did not start. "
            "Start it explicitly before using the Codex launcher."
        )

    http_port = get_container_env(CONTAINER_NAME, "SEMANTIC_MCP_HTTP_PORT") or "8011"
    base_url = f"http://127.0.0.1:{http_port}"
    wait_ready(base_url, READY_TIMEOUT_SEC)

    os.execvp(
        "docker",
        [
            "docker",
            "exec",
            "-i",
            "-e",
            f"SEMANTIC_MCP_PROXY_URL={base_url}/mcp",
            CONTAINER_NAME,
            "python3",
            "/repo/scripts/runtime/repo_semantic_stdio_proxy.py",
        ],
    )


if __name__ == "__main__":
    main()
