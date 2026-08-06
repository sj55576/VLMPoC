"""Frame source validation, threaded capture, and shutdown behaviour."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.vision.source import (
    FrameSourceError,
    MockFrameSource,
    OpenCVFrameSource,
    SourceSpec,
    create_frame_source,
    resolve_source_uri,
)


def _decodable_frames(path) -> int:
    """How many frames the encoder actually produced; it is not always what we wrote."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def _write_video(path, frames: int = 12) -> str:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, index % 3] = 255
        writer.write(frame)
    writer.release()
    if _decodable_frames(path) == 0:
        pytest.skip("this OpenCV build produced no decodable mp4v frames")
    return str(path)


def test_resolve_source_uri_defaults_and_validation(tmp_path):
    assert resolve_source_uri("mock", "anything", tmp_path) == ""
    assert resolve_source_uri("camera", "", tmp_path) == "0"
    assert resolve_source_uri("camera", "2", tmp_path) == "2"
    assert resolve_source_uri("rtsp", "rtsp://host/stream", tmp_path) == "rtsp://host/stream"
    with pytest.raises(FrameSourceError):
        resolve_source_uri("camera", "/dev/video0", tmp_path)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("rtsp", "file:///etc/passwd", tmp_path)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("nonsense", "", tmp_path)


def test_file_source_uri_is_confined_to_the_allowed_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    video = _write_video(root / "clip.mp4")
    assert resolve_source_uri("file", "clip.mp4", root) == str(video)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("file", "../outside.mp4", root)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("file", "/etc/passwd", root)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("file", "missing.mp4", root)
    with pytest.raises(FrameSourceError):
        resolve_source_uri("file", "", root)


def test_mock_source_produces_paced_frames():
    async def run():
        source = create_frame_source(SourceSpec(type="mock", target_fps=200))
        assert isinstance(source, MockFrameSource)
        await source.open()
        frame = await source.read()
        await source.read()
        await source.close()
        return frame, source.stats

    frame, stats = asyncio.run(run())
    assert frame.shape == (480, 640, 3)
    assert stats.frames_read == 2
    assert stats.opened is False


def test_file_source_reads_to_completion_then_reports_finished(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=8)
    expected = _decodable_frames(path)

    async def run():
        # The buffer is larger than the clip, so nothing can be dropped and the
        # consumer must see every captured frame.
        source = OpenCVFrameSource(SourceSpec(type="file", uri=path, target_fps=0, queue_size=32))
        await source.open()
        frames = []
        while (frame := await source.read()) is not None:
            frames.append(frame)
        await source.close()
        return frames, source.stats

    frames, stats = asyncio.run(run())
    assert len(frames) == expected
    assert stats.finished is True
    assert stats.frames_read == expected
    assert stats.frames_dropped == 0
    assert stats.last_error is None


def test_file_source_drops_frames_instead_of_growing_latency(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=40)
    expected = _decodable_frames(path)

    async def run():
        source = OpenCVFrameSource(SourceSpec(type="file", uri=path, target_fps=0, queue_size=2))
        await source.open()
        await asyncio.sleep(0.3)  # let the capture thread run ahead of the consumer
        while await source.read() is not None:
            await asyncio.sleep(0)
        await source.close()
        return source.stats

    stats = asyncio.run(run())
    assert stats.frames_read == expected
    assert stats.frames_dropped > 0


def test_unopenable_source_raises_instead_of_streaming_nothing(tmp_path):
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"definitely not a video")

    async def run():
        source = OpenCVFrameSource(SourceSpec(type="file", uri=str(broken), reconnect_seconds=0))
        with pytest.raises(FrameSourceError):
            await source.open()
        return source.stats

    stats = asyncio.run(run())
    assert stats.opened is False


def test_close_is_idempotent_and_stops_the_capture_thread(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", frames=200)

    async def run():
        source = OpenCVFrameSource(SourceSpec(type="file", uri=path, target_fps=0, queue_size=1))
        await source.open()
        await source.read()
        await source.close()
        await source.close()
        return source

    source = asyncio.run(run())
    assert source._thread is None
    assert source.stats.opened is False


def test_finished_is_not_a_drain_signal(tmp_path):
    """`finished` means the capture thread hit EOF, not that the consumer got the frames.

    A buffer larger than the clip can hold every frame at the moment `finished` is set,
    so waiting on it is not a valid completion signal for anything downstream.
    """
    path = _write_video(tmp_path / "clip.mp4", frames=10)
    expected = _decodable_frames(path)

    async def run():
        source = OpenCVFrameSource(SourceSpec(type="file", uri=path, target_fps=0, queue_size=256))
        await source.open()
        while not source.stats.finished:  # deliberately consume nothing while capturing
            await asyncio.sleep(0.01)
        buffered = len(source._buffer)
        delivered = 0
        while await source.read() is not None:
            delivered += 1
        await source.close()
        return buffered, delivered, source.stats

    buffered, delivered, stats = asyncio.run(run())
    assert stats.frames_read == expected
    assert buffered == expected, "every frame is still queued when finished is set"
    assert delivered == expected, "frames stay readable after the source reports finished"
