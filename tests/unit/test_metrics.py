"""Tests for the Prometheus text exposition renderer."""
from __future__ import annotations

from app.core.metrics import render_metrics


def test_empty_snapshot_still_renders_build_info() -> None:
    text = render_metrics({})

    assert text.endswith("\n")
    assert "# HELP vlmsop_build_info" in text
    assert "# TYPE vlmsop_build_info gauge" in text
    assert 'vlmsop_build_info{version="",mode="",vlm_provider=""} 1' in text
    # No other families should appear when their keys are absent.
    assert "vlmsop_session_active" not in text
    assert "vlmsop_frames_processed_total" not in text


def test_build_info_uses_provided_values() -> None:
    text = render_metrics({"version": "1.2.3", "mode": "full", "vlm_provider": "openai_compatible"})

    assert 'vlmsop_build_info{version="1.2.3",mode="full",vlm_provider="openai_compatible"} 1' in text


def test_booleans_render_as_one_or_zero() -> None:
    text = render_metrics({"session_active": True})
    assert "vlmsop_session_active 1" in text

    text = render_metrics({"session_active": False})
    assert "vlmsop_session_active 0" in text


def test_counters_get_total_suffix_without_doubling() -> None:
    snapshot = {
        "frames_processed": 42,
        "vlm_calls": 10,
        "vlm_failures": 2,
        "source_reconnects": 1,
        "sessions_total": 3,
        "vlm_total": 7,
    }
    text = render_metrics(snapshot)

    assert "# TYPE vlmsop_frames_processed_total counter" in text
    assert "vlmsop_frames_processed_total 42" in text
    assert "vlmsop_vlm_calls_total 10" in text
    assert "vlmsop_vlm_failures_total 2" in text
    assert "vlmsop_source_reconnects_total 1" in text
    # Already ends in _total: must not become sessions_total_total.
    assert "vlmsop_sessions_total 3" in text
    assert "vlmsop_sessions_total_total" not in text
    assert "vlmsop_vlm_total 7" in text
    assert "vlmsop_vlm_total_total" not in text


def test_gauges_are_unsuffixed() -> None:
    snapshot = {
        "websocket_subscribers": 5,
        "stream_subscribers": 2,
        "vlm_latency_ms_last": 123.4,
        "vlm_latency_ms_avg": 98.7,
        "sop_progress": 0.75,
    }
    text = render_metrics(snapshot)

    assert "# TYPE vlmsop_websocket_subscribers gauge" in text
    assert "vlmsop_websocket_subscribers 5" in text
    assert "vlmsop_stream_subscribers 2" in text
    assert "vlmsop_vlm_latency_ms_last 123.4" in text
    assert "vlmsop_vlm_latency_ms_avg 98.7" in text
    assert "vlmsop_sop_progress 0.75" in text


def test_events_by_type_labelled_family() -> None:
    text = render_metrics({"events_by_type": {"step_completed": 4, "sop_violation": 1}})

    assert "# TYPE vlmsop_events_by_type counter" in text
    assert 'vlmsop_events_by_type{event_type="sop_violation"} 1' in text
    assert 'vlmsop_events_by_type{event_type="step_completed"} 4' in text


def test_events_total_and_breakdown_are_separate_families() -> None:
    text = render_metrics({"events_total": 5, "events_by_type": {"step_completed": 5}})

    # The scalar total must not share a metric name with the labelled breakdown.
    assert text.count("# TYPE vlmsop_events_total counter") == 1
    assert "vlmsop_events_total 5" in text
    assert "vlmsop_events_total{" not in text
    assert 'vlmsop_events_by_type{event_type="step_completed"} 5' in text


def test_events_by_severity_labelled_family() -> None:
    text = render_metrics({"events_by_severity": {"WARN": 2, "INFO": 9}})

    assert "# TYPE vlmsop_events_by_severity counter" in text
    assert 'vlmsop_events_by_severity{severity="INFO"} 9' in text
    assert 'vlmsop_events_by_severity{severity="WARN"} 2' in text


def test_label_values_are_escaped() -> None:
    text = render_metrics({"events_by_type": {'weird"type\nwith\\backslash': 1}})

    assert 'event_type="weird\\"type\\nwith\\\\backslash"' in text


def test_output_is_deterministic_regardless_of_dict_order() -> None:
    snapshot_a = {"events_by_type": {"b": 1, "a": 2}, "events_by_severity": {"WARN": 1, "INFO": 2}}
    snapshot_b = {"events_by_severity": {"INFO": 2, "WARN": 1}, "events_by_type": {"a": 2, "b": 1}}

    assert render_metrics(snapshot_a) == render_metrics(snapshot_b)


def test_unknown_keys_are_ignored_without_raising() -> None:
    text = render_metrics({"totally_unknown_key": 123, "version": "x"})

    assert "totally_unknown_key" not in text
    assert 'version="x"' in text


def test_missing_keys_never_raise() -> None:
    # Should not raise for a completely bare snapshot, nor for a partially filled one.
    render_metrics({})
    render_metrics({"frames_processed": 1})
    render_metrics({"events_by_type": {}})
    render_metrics({"events_by_severity": {}})
