"""SQLite connection helpers shared by storage adapters."""

from __future__ import annotations

from pathlib import Path
import sqlite3


class ClosingConnection(sqlite3.Connection):
    """sqlite3 connection that closes when used as a context manager."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect_row_factory(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection that returns rows and closes on context exit."""

    connection = sqlite3.connect(db_path, timeout=30.0, factory=ClosingConnection)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.row_factory = sqlite3.Row
    return connection
