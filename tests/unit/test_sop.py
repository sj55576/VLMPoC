from datetime import datetime, timedelta, timezone
from pathlib import Path
from app.sop.conditions import ConditionEvaluator
from app.sop.loader import load_sop
from app.sop.models import SOPDefinition, StepDefinition, SOPMeta
from app.sop.state_machine import SOPStateMachine
from app.vision.models import Detection, Observation

def test_loader_and_boolean_conditions():
    sop=load_sop(Path("sop/example_assembly.yaml")); assert len(sop.steps)==4
    obs=Observation(objects=[Detection(class_name="person",confidence=.9,bbox=(0,0,10,10))])
    evaluator=ConditionEvaluator({})
    assert evaluator.evaluate({"all":[{"type":"object_present","object":"person"},{"not":{"type":"object_present","object":"helmet"}}]},obs,set()).passed
    assert evaluator.evaluate({"any":[{"type":"object_present","object":"helmet"},{"type":"object_present","object":"person"}]},obs,set()).passed

def test_state_duration_and_timeout():
    sop=SOPDefinition(sop=SOPMeta(id="x",name="x",version="1"),steps=[StepDefinition(id="one",name="one",conditions={"type":"object_present","object":"x"},minimum_duration_seconds=1,timeout_seconds=2)])
    state=SOPStateMachine(sop); now=datetime.now(timezone.utc)
    from app.sop.models import ConditionResult, StepStatus
    ok=ConditionResult(condition_id="x",type="x",passed=True,confidence=1,reason="ok")
    state.update(ok,now); assert state.current_runtime.status==StepStatus.ACTIVE
    state.update(ok,now+timedelta(seconds=1.1)); assert state.steps["one"].status==StepStatus.COMPLETED
    timeout=SOPStateMachine(sop); timeout.update(ConditionResult(condition_id="x",type="x",passed=False,confidence=0,reason="no"),now); timeout.update(ConditionResult(condition_id="x",type="x",passed=False,confidence=0,reason="no"),now+timedelta(seconds=2.1)); assert timeout.steps["one"].status==StepStatus.FAILED
