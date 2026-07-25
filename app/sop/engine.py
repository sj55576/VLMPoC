"""SOP rule engine facade."""
from datetime import datetime

from app.vision.models import Observation

from .conditions import ConditionEvaluator
from .models import ConditionResult, SOPDefinition
from .state_machine import SOPStateMachine


class SOPEngine:
    def __init__(self, sop: SOPDefinition) -> None:
        self.sop, self.evaluator, self.state = sop, ConditionEvaluator(sop.regions), SOPStateMachine(sop)

    def evaluate(self, observation: Observation, now: datetime) -> tuple[ConditionResult | None, str | None]:
        step = self.state.current
        if not step: return None, None
        result = self.evaluator.evaluate(step.conditions, observation, self.state.completed_ids())
        _, event = self.state.update(result, now)
        return result, event
