from __future__ import annotations

import unittest

from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.indexing.embedding_pipeline import embed_chunks
from services.repo_semantic.models import ChunkRecord


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def index_profile(self) -> str:
        return "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]

    def backend_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"

    def healthcheck(self) -> None:
        return None

    def endpoint_label(self) -> str:
        return "fake"


def _chunk(point_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        point_id=point_id,
        scope="code",
        relative_path=f"src/{point_id}.py",
        language="python",
        chunk_type="python_function",
        text=text,
        start_line=1,
        end_line=1,
        content_hash=point_id,
        source_mtime=0.0,
    )


class EmbeddingPipelineTests(unittest.TestCase):
    def test_embed_chunks_batches_by_docs_and_chars(self) -> None:
        provider = FakeEmbeddingProvider()
        chunks = [
            _chunk("a", "aaa"),
            _chunk("b", "bbbb"),
            _chunk("c", "cc"),
        ]

        vectors = embed_chunks(
            chunks=chunks,
            scope="code",
            embedding_provider=provider,
            max_batch_docs=2,
            max_batch_chars=6,
        )

        self.assertEqual(provider.calls, [["aaa"], ["bbbb", "cc"]])
        self.assertEqual(vectors, [[3.0], [4.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
