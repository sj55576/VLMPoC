"""A cached VLM verdict must expire instead of silently confirming later steps."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.sop.conditions import ConditionEvaluator
from app.vision.models import Observation

CONFIRMED = {"step_status": "IN_PROGRESS", "confidence": 0.9, "scene_summary": "tightening", "current_step_id": None}
CONDITION = {"type": "vlm_confirmation", "question": "is the worker tightening the part"}


def _observation(age: float | None) -> Observation:
    return Observation(timestamp=datetime.now(UTC), vlm_result=CONFIRMED, vlm_result_age_seconds=age)


def test_fresh_result_confirms_the_step():
    result = ConditionEvaluator({}, 15.0).evaluate(CONDITION, _observation(2.0), set())
    assert result.passed is True
    assert result.evidence["stale"] is False


def test_stale_result_no_longer_confirms_the_step():
    result = ConditionEvaluator({}, 15.0).evaluate(CONDITION, _observation(60.0), set())
    assert result.passed is False
    assert result.confidence == 0.0
    assert result.evidence["stale"] is True
    assert "stale" in result.reason


def test_condition_can_override_the_configured_budget():
    observation = _observation(20.0)
    assert ConditionEvaluator({}, 15.0).evaluate({**CONDITION, "max_age_seconds": 30}, observation, set()).passed is True
    assert ConditionEvaluator({}, 15.0).evaluate({**CONDITION, "max_age_seconds": 5}, observation, set()).passed is False


def test_zero_budget_disables_the_staleness_check():
    result = ConditionEvaluator({}, 0.0).evaluate(CONDITION, _observation(3600.0), set())
    assert result.passed is True


def test_missing_vlm_result_is_not_reported_as_stale():
    result = ConditionEvaluator({}, 15.0).evaluate(CONDITION, Observation(timestamp=datetime.now(UTC)), set())
    assert result.passed is False
    assert result.evidence["stale"] is False


def test_pipeline_reports_result_age(monkeypatch):
    from app.core.config import VLMSettings
    from app.sop.engine import SOPEngine
    from app.sop.loader import load_sop
    from app.vision.detector import MockDetector
    from app.vision.pipeline import VisionPipeline
    from app.vision.pose import MockPoseEstimator
    from app.vision.tracker import IoUTracker
    from app.vlm.mock_provider import MockVLMProvider

    sop = load_sop(Path("sop/example_assembly.yaml"))
    pipeline = VisionPipeline(MockDetector(), MockPoseEstimator(), IoUTracker(), SOPEngine(sop), MockVLMProvider(), vlm_settings=VLMSettings())
    base = datetime.now(UTC)
    assert pipeline.vlm_result_age(base) is None
    pipeline.last_vlm = {"step_status": "UNKNOWN"}
    pipeline.last_vlm_result_at = base
    assert pipeline.vlm_result_age(base + timedelta(seconds=7)) == 7.0
