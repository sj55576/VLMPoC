"""Monotonic step state machine with duration and timeout semantics."""
from __future__ import annotations

from datetime import datetime

from .models import ConditionResult, SOPDefinition, StepRuntime, StepStatus


class SOPStateMachine:
    def __init__(self, sop: SOPDefinition) -> None:
        self.sop, self.index = sop, 0
        self._index_by_id = {step.id: index for index, step in enumerate(sop.steps)}
        self.steps = {step.id: StepRuntime(step_id=step.id) for step in sop.steps}

    @property
    def current(self): return self.sop.steps[self.index] if self.index < len(self.sop.steps) else None
    @property
    def current_runtime(self): return self.steps[self.current.id] if self.current else None

    def update(self, result: ConditionResult, now: datetime) -> tuple[StepRuntime | None, str | None]:
        """Advance only forward after sustained truth; return a transition event when it occurs."""
        step = self.current
        if not step: return None, None
        runtime = self.steps[step.id]
        if runtime.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}:
            return runtime, None
        if runtime.status == StepStatus.PENDING:
            runtime.status, runtime.started_at = StepStatus.ACTIVE, now
        if step.timeout_seconds is not None and (now - runtime.started_at).total_seconds() >= step.timeout_seconds:
            runtime.status, runtime.reason = StepStatus.FAILED, "step timeout"
            return runtime, "step_failed"
        runtime.confidence, runtime.reason = result.confidence, result.reason
        if result.passed:
            runtime.condition_true_since = runtime.condition_true_since or now
            if (now-runtime.condition_true_since).total_seconds() >= step.minimum_duration_seconds:
                runtime.status, runtime.completed_at = StepStatus.COMPLETED, now
                self._advance(step)
                return runtime, "step_completed"
        else:
            runtime.condition_true_since = None
        return runtime, None

    def _advance(self, step) -> None:
        """Follow explicit SOP edges and mark bypassed linear steps as skipped."""
        if step.terminal:
            self.index = len(self.sop.steps)
            return
        next_index = self._index_by_id.get(step.on_success, self.index + 1)
        if next_index > self.index + 1:
            for skipped in self.sop.steps[self.index + 1:next_index]:
                self.steps[skipped.id].status = StepStatus.SKIPPED
                self.steps[skipped.id].reason = f"bypassed by on_success from {step.id}"
        self.index = next_index

    def completed_ids(self) -> set[str]: return {key for key, value in self.steps.items() if value.status == StepStatus.COMPLETED}

    def progress(self) -> float: return len(self.completed_ids()) / max(1, len(self.sop.steps))
