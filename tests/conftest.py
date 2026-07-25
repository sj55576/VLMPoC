"""Test isolation: every test run gets its own SQLite file instead of ./data/app.db."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Point the application at a throwaway database so tests never touch developer data."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("FRAME_STORAGE_DIR", str(tmp_path / "frames"))
    yield
