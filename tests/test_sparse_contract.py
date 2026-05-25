from __future__ import annotations

import unittest

from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.sparse import (
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    analyze_exact_anchors,
    build_sparse_manifest,
    encode_sparse_document,
    encode_sparse_query,
    sparse_unique_terms,
    sparse_terms,
)


def _chunk(chunk_id: str, relative_path: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        point_id=chunk_id,
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="generic",
        text=text,
        start_line=1,
        end_line=1,
        content_hash=chunk_id,
        source_mtime=0.0,
    )


class SparseContractTests(unittest.TestCase):
    def test_code_aware_analyzer_preserves_exact_anchors_and_expansions(self) -> None:
        cases = {
            "DATABASE_URL": {"database_url", "database", "url"},
            "UserRepository.findByEmail": {
                "userrepository.findbyemail",
                "userrepository",
                "findbyemail",
                "user",
                "repository",
                "find",
                "by",
                "email",
            },
            "services/user/api/connect_flow.py": {
                "services/user/api/connect_flow.py",
                "services",
                "user",
                "api",
                "connect_flow",
                "connect",
                "flow",
                "py",
            },
            "/api/v1/users/{id}": {"api/v1/users/{id}", "api", "v1", "users", "id"},
            "--target-repo-path": {"target-repo-path", "target", "repo", "path"},
            "permission denied": {"permission", "denied"},
        }
        for raw, expected_terms in cases.items():
            with self.subTest(raw=raw):
                self.assertTrue(expected_terms.issubset(set(sparse_unique_terms(raw))))

    def test_manifest_is_deterministic_across_input_order(self) -> None:
        chunks = [
            _chunk("b", "src/b.py", "DATABASE_URL database"),
            _chunk("a", "src/a.py", "connect_flow database"),
        ]

        first = build_sparse_manifest(scope="code", chunks=chunks, schema_version=2)
        second = build_sparse_manifest(scope="code", chunks=list(reversed(chunks)), schema_version=2)

        self.assertEqual(first.lexical_analyzer_version, LEXICAL_ANALYZER_VERSION)
        self.assertEqual(first.sparse_encoder_kind, SPARSE_ENCODER_KIND)
        self.assertEqual(first.token_to_id, second.token_to_id)
        self.assertEqual(first.vocabulary_hash, second.vocabulary_hash)
        self.assertEqual(first.corpus_stats_hash, second.corpus_stats_hash)
        self.assertEqual(first.manifest_content_hash, second.manifest_content_hash)

    def test_sparse_query_and_document_vectors_share_manifest_vocabulary(self) -> None:
        chunks = [_chunk("a", "src/a.py", "DATABASE_URL database settings")]
        manifest = build_sparse_manifest(scope="code", chunks=chunks, schema_version=2)

        document_vector = encode_sparse_document(chunks[0].text, manifest)
        query_vector = encode_sparse_query("DATABASE_URL", manifest)

        self.assertTrue(document_vector.indices)
        self.assertTrue(query_vector.indices)
        self.assertTrue(set(query_vector.indices).issubset(set(document_vector.indices)))

    def test_sparse_terms_preserve_repeated_source_span_frequency(self) -> None:
        terms = sparse_terms("alpha alpha alpha")

        self.assertEqual(terms.count("alpha"), 3)

    def test_exact_anchor_analysis_returns_safe_rg_argv_hints(self) -> None:
        analysis = analyze_exact_anchors('where is DATABASE_URL in "/api/v1/users/{id}"')
        payload = analysis.model_dump()

        self.assertTrue(payload["exact_anchor_heavy"])
        self.assertEqual(payload["anchors"][0]["surface"], "DATABASE_URL")
        self.assertEqual(payload["anchors"][0]["anchor_type"], "env_var")
        self.assertEqual(payload["argv_hints"][0][0], "rg")
        self.assertIn("-e", payload["argv_hints"][0])
        self.assertEqual(payload["cwd_hint"], "repo_root")

    def test_exact_anchor_analysis_classifies_files_symbols_and_sql(self) -> None:
        file_payload = analyze_exact_anchors("open app.py").model_dump()
        symbol_payload = analyze_exact_anchors("find UserRepository.findByEmail").model_dump()
        sql_payload = analyze_exact_anchors("table public.users").model_dump()

        self.assertEqual(file_payload["anchors"][0]["anchor_type"], "file_name")
        self.assertEqual(symbol_payload["anchors"][0]["anchor_type"], "symbol")
        self.assertEqual(sql_payload["anchors"][0]["anchor_type"], "sql_identifier")


if __name__ == "__main__":
    unittest.main()
