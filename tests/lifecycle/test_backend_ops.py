from __future__ import annotations

from types import SimpleNamespace
import unittest

from services.repo_semantic.lifecycle.backend_ops import (
    backend_action_plan_payload,
    backend_status_payload,
    embedding_backend_switch_payload,
    resolve_backend_entry,
    runtime_embedding_backend_id,
)
from services.repo_semantic.models import BackendRegistryEntry


def _backend(backend_id: str, *, role: str = "embedding", launch_mode: str = "external_manual") -> BackendRegistryEntry:
    return BackendRegistryEntry(
        backend_id=backend_id,
        role=role,
        backend_type="embedding",
        transport="http",
        endpoint=None,
        health_path="/health",
        location_type="host",
        device_type="cpu",
        model_name="fake-model",
        config_blob={"launch_mode": launch_mode},
        created_at="now",
        updated_at="now",
    )


class FakeRegistry:
    def __init__(self, selected: BackendRegistryEntry) -> None:
        self.selected = selected

    def get_role_backend(self, _role: str) -> BackendRegistryEntry:
        return self.selected

    def get_backend(self, backend_id: str) -> BackendRegistryEntry | None:
        return self.selected if self.selected.backend_id == backend_id else None


class BackendOpsTests(unittest.TestCase):
    def test_embedding_backend_switch_payload_reports_selected_runtime_mismatch(self) -> None:
        selected = _backend("remote-e5")
        runtime = SimpleNamespace(
            registry=FakeRegistry(selected),
            indexer=SimpleNamespace(_settings=SimpleNamespace(embedding_backend_id="local-e5")),
            search_service=SimpleNamespace(index_status=lambda: SimpleNamespace(model_dump=lambda: {"ready": True})),
        )

        payload = embedding_backend_switch_payload(runtime)

        self.assertEqual(runtime_embedding_backend_id(runtime), "local-e5")
        self.assertEqual(payload["requested_backend_id"], "remote-e5")
        self.assertEqual(payload["runtime_backend_id"], "local-e5")
        self.assertTrue(payload["backend_switch_required"])

    def test_backend_status_and_action_plan_are_operational_payloads(self) -> None:
        selected = _backend("remote-e5")
        runtime = SimpleNamespace(
            registry=FakeRegistry(selected),
            indexer=SimpleNamespace(_settings=SimpleNamespace(embedding_backend_id="local-e5")),
            embedding_provider=SimpleNamespace(healthcheck=lambda: None),
        )

        status = backend_status_payload(runtime, selected)
        plan = backend_action_plan_payload(runtime, selected, target_repo_root="C:/repo")

        self.assertTrue(status["runtime_switch_required"])
        self.assertEqual(plan["operator_action_state"], "manual_backend_start_required")
        self.assertEqual(plan["repo_root"], "C:/repo")
        self.assertTrue(plan["steps"])

    def test_resolve_backend_entry_requires_role_or_id(self) -> None:
        runtime = SimpleNamespace(registry=FakeRegistry(_backend("remote-e5")))

        self.assertEqual(resolve_backend_entry(runtime, None, "remote-e5").backend_id, "remote-e5")
        with self.assertRaisesRegex(RuntimeError, "Either role or backend_id"):
            resolve_backend_entry(runtime, None, None)


if __name__ == "__main__":
    unittest.main()
