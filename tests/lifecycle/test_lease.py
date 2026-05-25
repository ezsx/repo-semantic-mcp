from __future__ import annotations

from types import SimpleNamespace
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.lifecycle.lease import _start_lease_heartbeat, lifecycle_global_mutation
from services.repo_semantic.registry import LifecycleLease, RepoRegistry


class FakeHeartbeatRegistry:
    def __init__(self) -> None:
        self.calls = 0

    def heartbeat_lifecycle_lease(self, lease: LifecycleLease, *, ttl_sec: int) -> bool:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("database is locked")
        return False


class LifecycleLeaseTests(unittest.TestCase):
    def test_heartbeat_survives_transient_registry_error(self) -> None:
        registry = FakeHeartbeatRegistry()
        lease = LifecycleLease(
            lease_key="repo|profile|backend",
            lease_id="lease",
            audit_id="audit",
            repo_root="repo",
            index_profile="profile",
            backend_id="backend",
            operation="rebuild_index",
            owner_id="owner",
            idempotency_key=None,
            acquired_at="2026-05-22T00:00:00+00:00",
            heartbeat_at="2026-05-22T00:00:00+00:00",
            expires_at="2026-05-22T00:15:00+00:00",
        )

        stop_event, thread = _start_lease_heartbeat(
            registry,
            lease,
            ttl_sec=900,
            interval_sec=0.01,
        )
        try:
            deadline = time.monotonic() + 1.0
            while registry.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            stop_event.set()
            thread.join(timeout=1)

        self.assertGreaterEqual(registry.calls, 2)

    def test_global_lifecycle_mutation_blocks_across_runtime_repos(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
            runtime_a = SimpleNamespace(registry=registry)
            runtime_b = SimpleNamespace(registry=registry)

            with lifecycle_global_mutation(runtime_a, "set_role_backend", scope="backend_roles"):
                with self.assertRaises(RuntimeError):
                    with lifecycle_global_mutation(runtime_b, "set_role_backend", scope="backend_roles"):
                        pass

            audit_events = registry.list_lifecycle_audit(limit=1)
            self.assertEqual(audit_events[0]["operation"], "set_role_backend")
            self.assertEqual(audit_events[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
