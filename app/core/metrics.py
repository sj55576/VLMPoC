"""Render an application state snapshot as Prometheus text exposition format v0.0.4."""
from __future__ import annotations

from typing import Any

_PREFIX = "vlmsop_"


def _escape(value: str) -> str:
    """Escape a label value per the Prometheus text format spec."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: Any) -> str:
    """Render a metric sample value: bools as 1/0, everything else via str()."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _counter_name(key: str) -> str:
    """Counter metric names are suffixed with _total, without doubling an existing suffix."""
    return key if key.endswith("_total") else f"{key}_total"


def _header(lines: list[str], name: str, metric_type: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")


def render_metrics(snapshot: dict[str, Any]) -> str:
    """Render a Prometheus text-format exposition of the given snapshot.

    `snapshot` keys are all optional; missing keys are tolerated (never raise) and
    simply produce no sample for that metric, except build_info which always renders
    using empty-string defaults. Families are emitted in a fixed order with sorted
    label keys, so output is deterministic for a given snapshot.
    """
    lines: list[str] = []

    name = f"{_PREFIX}build_info"
    _header(lines, name, "gauge", "Static build/deployment information.")
    version = _escape(str(snapshot.get("version", "")))
    mode = _escape(str(snapshot.get("mode", "")))
    vlm_provider = _escape(str(snapshot.get("vlm_provider", "")))
    lines.append(f'{name}{{version="{version}",mode="{mode}",vlm_provider="{vlm_provider}"}} 1')

    if "session_active" in snapshot:
        name = f"{_PREFIX}session_active"
        _header(lines, name, "gauge", "Whether a monitoring session is currently active.")
        lines.append(f"{name} {_fmt(snapshot['session_active'])}")

    counter_specs = [
        ("frames_processed", "Total number of frames processed."),
        ("vlm_calls", "Total number of VLM inference calls made."),
        ("vlm_failures", "Total number of failed VLM inference calls."),
        ("source_reconnects", "Total number of source reconnect attempts."),
        ("sessions_total", "Total number of sessions started."),
        ("vlm_total", "Total number of VLM results recorded."),
    ]
    for key, help_text in counter_specs:
        if key in snapshot:
            metric_name = f"{_PREFIX}{_counter_name(key)}"
            _header(lines, metric_name, "counter", help_text)
            lines.append(f"{metric_name} {_fmt(snapshot[key])}")

    gauge_specs = [
        ("websocket_subscribers", "Current number of websocket subscribers."),
        ("stream_subscribers", "Current number of live stream subscribers."),
        ("vlm_latency_ms_last", "Latency in milliseconds of the most recent VLM call."),
        ("vlm_latency_ms_avg", "Average latency in milliseconds of VLM calls."),
        ("sop_progress", "Current SOP completion progress, from 0 to 1."),
    ]
    for key, help_text in gauge_specs:
        if key in snapshot:
            metric_name = f"{_PREFIX}{key}"
            _header(lines, metric_name, "gauge", help_text)
            lines.append(f"{metric_name} {_fmt(snapshot[key])}")

    events_total_present = "events_total" in snapshot
    events_by_type = snapshot.get("events_by_type") or {}
    if events_total_present or events_by_type:
        name = f"{_PREFIX}events_total"
        _header(lines, name, "counter", "Total number of SOP/activity events recorded.")
        if events_total_present:
            lines.append(f"{name} {_fmt(snapshot['events_total'])}")
        for key in sorted(events_by_type):
            lines.append(f'{name}{{event_type="{_escape(str(key))}"}} {_fmt(events_by_type[key])}')

    events_by_severity = snapshot.get("events_by_severity") or {}
    if events_by_severity:
        name = f"{_PREFIX}events_by_severity"
        _header(lines, name, "counter", "Total number of events by severity.")
        for key in sorted(events_by_severity):
            lines.append(f'{name}{{severity="{_escape(str(key))}"}} {_fmt(events_by_severity[key])}')

    return "\n".join(lines) + "\n"
