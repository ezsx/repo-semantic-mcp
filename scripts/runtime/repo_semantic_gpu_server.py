"""Минимальный WSL-native embedding server для repo-semantic-search.

Сервер поднимает только то, что нужно semantic MCP:
- `/health`
- `/embed`
- `/v1/embeddings`

Он intentionally не тянет reranker/NLI/ColBERT, потому что для repo-semantic-mcp
нужен только dense embedding backend. Это делает GPU path самодостаточным и
убирает зависимость от соседнего `rag_app`.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger("repo_semantic_gpu_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SERVER_HOST = os.environ.get("REPO_SEMANTIC_GPU_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("REPO_SEMANTIC_GPU_SERVER_PORT", "8084"))
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/mnt/c/llms/models/pplx-embed-v1-0.6B",
)
EMBEDDING_MODEL_NAME = os.environ.get(
    "REPO_SEMANTIC_GPU_SERVER_MODEL_NAME",
    "pplx-embed-v1",
)
MAX_LENGTH = int(os.environ.get("REPO_SEMANTIC_GPU_SERVER_MAX_LENGTH", "4096"))

_TOKENIZER = None
_MODEL = None


class ReusableHttpServer(HTTPServer):
    """HTTP server с `SO_REUSEADDR`, чтобы повторный старт не залипал на порту."""

    allow_reuse_address = True


def _load_model() -> None:
    """Загрузить embedding model в GPU-память один раз при старте процесса."""

    global _TOKENIZER, _MODEL

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for repo semantic GPU server")

    LOGGER.info("loading embedding model from %s", EMBEDDING_MODEL_PATH)
    _TOKENIZER = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_PATH,
        trust_remote_code=True,
    )
    # pplx-embed использует bf16 стабильнее fp16 на длинных текстах.
    _MODEL = AutoModel.from_pretrained(
        EMBEDDING_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    LOGGER.info("embedding model is ready")


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Построить L2-normalized embeddings для списка текстов."""

    if _TOKENIZER is None or _MODEL is None:
        raise RuntimeError("embedding model is not loaded")

    filtered = [text for text in texts if isinstance(text, str) and text.strip()]
    if not filtered:
        return []

    with torch.no_grad():
        encoded = _TOKENIZER(
            filtered,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")
        outputs = _MODEL(**encoded)
        hidden = outputs.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (hidden * mask).sum(1) / mask.sum(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    """HTTP API, совместимый с тем, что ожидает `TeiProvider`."""

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        """Не дублировать stdlib access logs поверх общего logger."""

        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: object, status: int = 200) -> None:
        """Вернуть JSON payload клиенту."""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        """Прочитать JSON request body."""

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8", errors="replace"))

    def do_GET(self) -> None:  # noqa: N802
        """Обслужить liveness/health endpoint."""

        if self.path == "/health":
            status = "ok" if _MODEL is not None else "loading"
            self._send_json(
                {
                    "status": status,
                    "model": EMBEDDING_MODEL_NAME,
                    "model_path": EMBEDDING_MODEL_PATH,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
            )
            return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        """Обслужить embedding endpoints."""

        try:
            payload = self._read_json_body()
            if self.path == "/embed":
                texts = payload.get("inputs", payload.get("texts", []))
                if isinstance(texts, str):
                    texts = [texts]
                self._send_json(_embed_texts(texts))
                return
            if self.path == "/v1/embeddings":
                texts = payload.get("input", payload.get("inputs", []))
                if isinstance(texts, str):
                    texts = [texts]
                vectors = _embed_texts(texts)
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": index, "embedding": vector}
                            for index, vector in enumerate(vectors)
                        ],
                        "model": EMBEDDING_MODEL_NAME,
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    }
                )
                return
            self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("request failed for path %s", self.path)
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    """Запустить embedding server после eager model load."""

    _load_model()
    server = ReusableHttpServer((SERVER_HOST, SERVER_PORT), Handler)
    LOGGER.info(
        "repo semantic GPU embedding server is listening on %s:%s",
        SERVER_HOST,
        SERVER_PORT,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
