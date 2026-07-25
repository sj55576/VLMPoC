"""Tests for FrameStore: save/read round-trip, traversal safety, and retention purging."""
from __future__ import annotations

import time

import numpy as np

from app.storage.frames import FrameStore


def _frame(h: int = 40, w: int = 60) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_save_and_read_round_trip(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")

    relative = store.save("session-1", 42, _frame())

    assert relative == "session-1/42.jpg"
    resolved = store.path(relative)
    assert resolved is not None
    assert resolved.is_file()
    assert resolved.read_bytes()[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_save_downscales_to_max_dim(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames", max_dim=32)

    relative = store.save("session-1", 1, _frame(h=64, w=128))
    assert relative is not None

    import cv2
    decoded = cv2.imread(str(store.path(relative)))
    assert max(decoded.shape[:2]) <= 32


def test_save_sanitizes_session_and_event_ids(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")

    relative = store.save("../../evil session!", 7, _frame())

    assert relative is not None
    assert ".." not in relative
    assert " " not in relative
    assert "!" not in relative


def test_save_returns_none_for_missing_frame(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    assert store.save("session-1", 1, None) is None


def test_disabled_store_returns_none_and_creates_nothing(tmp_path) -> None:
    directory = tmp_path / "frames"
    store = FrameStore(directory, enabled=False)

    assert store.enabled is False
    assert store.save("session-1", 1, _frame()) is None
    assert not directory.exists()


def test_path_rejects_relative_traversal(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    store.save("session-1", 1, _frame())

    assert store.path("../../etc/passwd") is None
    assert store.path("session-1/../../../etc/passwd") is None


def test_path_rejects_absolute_paths(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not a real jpeg")

    assert store.path(str(outside)) is None


def test_path_returns_none_for_missing_file(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    assert store.path("session-1/999.jpg") is None


def test_purge_older_than_deletes_old_files_and_empty_dirs(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    old_relative = store.save("session-old", 1, _frame())
    new_relative = store.save("session-new", 2, _frame())
    assert old_relative and new_relative

    old_path = store.path(old_relative)
    old_time = time.time() - 40 * 86400
    import os
    os.utime(old_path, (old_time, old_time))

    deleted = store.purge_older_than(30)

    assert deleted == 1
    assert store.path(old_relative) is None
    assert not (tmp_path / "frames" / "session-old").exists()
    assert store.path(new_relative) is not None


def test_purge_older_than_zero_is_noop(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    relative = store.save("session-1", 1, _frame())
    assert store.purge_older_than(0) == 0
    assert store.path(relative) is not None


def test_purge_session_removes_only_that_session(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    a = store.save("session-a", 1, _frame())
    b = store.save("session-b", 1, _frame())

    deleted = store.purge_session("session-a")

    assert deleted == 1
    assert store.path(a) is None
    assert store.path(b) is not None
