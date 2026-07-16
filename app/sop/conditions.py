"""Recursive SOP condition evaluator with evidence-rich leaf results."""
from __future__ import annotations
from typing import Any
from app.vision.geometry import calculate_joint_angle, distance, inside_region
from app.vision.models import Observation
from .models import ConditionResult


class ConditionEvaluator:
    def __init__(self, regions: dict[str, dict[str, float]]) -> None:
        self.regions = regions

    def evaluate(self, expression: dict[str, Any], observation: Observation, completed_steps: set[str]) -> ConditionResult:
        """Evaluate all/any/not recursively, or a supported leaf condition."""
        for operator in ("all", "any"):
            if operator in expression:
                results = [self.evaluate(x, observation, completed_steps) for x in expression[operator]]
                passed = all(x.passed for x in results) if operator == "all" else any(x.passed for x in results)
                confidence = (min((x.confidence for x in results), default=0.0) if operator == "all" else max((x.confidence for x in results), default=0.0))
                return ConditionResult(condition_id=operator, type=operator, passed=passed, confidence=confidence, reason=f"{operator}({', '.join(x.reason for x in results)})", evidence={"children": [x.model_dump() for x in results]})
        if "not" in expression:
            child = self.evaluate(expression["not"], observation, completed_steps)
            return ConditionResult(condition_id="not", type="not", passed=not child.passed, confidence=child.confidence, reason=f"not ({child.reason})", evidence={"child": child.model_dump()})
        return self._leaf(expression, observation, completed_steps)

    def _objects(self, observation: Observation, name: str):
        return [o for o in observation.objects if o.class_name == name]

    def _result(self, expression: dict[str, Any], passed: bool, confidence: float, reason: str, evidence: dict[str, Any] | None = None) -> ConditionResult:
        return ConditionResult(condition_id=expression.get("id", expression["type"]), type=expression["type"], passed=passed, confidence=confidence, reason=reason, evidence=evidence or {})

    def _leaf(self, c: dict[str, Any], obs: Observation, completed: set[str]) -> ConditionResult:
        type_ = c["type"]
        if type_ in {"object_present", "object_absent", "object_count"}:
            found = self._objects(obs, c["object"]); count = len(found)
            passed = count > 0 if type_ == "object_present" else count == 0 if type_ == "object_absent" else count == c.get("count", 1)
            return self._result(c, passed, max((x.confidence for x in found), default=.0), f"{c['object']} count={count}", {"track_ids": [x.track_id for x in found], "frame_id": obs.frame_id})
        if type_ == "object_inside_region":
            region = self.regions[c["region"]]; matches = self._objects(obs, c["object"])
            for item in matches:
                center = item.center(); normal = (center[0]/obs.width, center[1]/obs.height)
                if inside_region(normal, region): return self._result(c, True, item.confidence, f"{c['object']} is inside {c['region']}", {"track_id": item.track_id, "bbox": item.bbox, "frame_id": obs.frame_id})
            return self._result(c, False, 0, f"{c['object']} is not inside {c['region']}")
        if type_ == "object_near_object":
            return self._near_objects(c, obs, c["object"], c["target_object"])
        if type_ == "object_near_body_part":
            objects = self._objects(obs, c["object"])
            poses = [p for p in obs.poses if c["body_part"] in p.keypoints]
            for obj in objects:
                point = (obj.center()[0]/obs.width, obj.center()[1]/obs.height)
                for pose in poses:
                    kp = pose.keypoints[c["body_part"]]
                    d = distance(point, (kp.x, kp.y))
                    if d <= c["max_distance"]: return self._result(c, True, min(obj.confidence, kp.confidence), f"{c['object']} near {c['body_part']} ({d:.3f})", {"track_id": obj.track_id, "frame_id": obs.frame_id})
            return self._result(c, False, 0, f"{c['object']} not near {c['body_part']}")
        if type_ == "body_part_inside_region":
            region = self.regions[c["region"]]
            for pose in obs.poses:
                kp = pose.keypoints.get(c["body_part"])
                if kp and inside_region((kp.x, kp.y), region): return self._result(c, True, kp.confidence, f"{c['body_part']} inside {c['region']}")
            return self._result(c, False, 0, f"{c['body_part']} outside {c['region']}")
        if type_ == "pose_angle":
            for pose in obs.poses:
                keys = [pose.keypoints.get(x) for x in (c["point_a"], c["point_b"], c["point_c"])]
                if all(keys):
                    angle = calculate_joint_angle(*keys)
                    passed = c.get("min_degrees", 0) <= angle <= c.get("max_degrees", 180)
                    return self._result(c, passed, min(x.confidence for x in keys), f"joint angle={angle:.1f}", {"angle": angle})
            return self._result(c, False, 0, "required keypoints absent")
        if type_ == "hand_object_interaction":
            # Tool must be near both a wrist and target object.
            tool = c["hand_object"]; target = c["target_object"]
            wrist_expr = {"type": "object_near_body_part", "object": tool, "body_part": "left_wrist", "max_distance": c["max_distance"]}
            left = self._leaf(wrist_expr, obs, completed)
            wrist_expr["body_part"] = "right_wrist"; right = self._leaf(wrist_expr, obs, completed)
            near = self._near_objects({**c, "type": "object_near_object"}, obs, tool, target)
            passed = (left.passed or right.passed) and near.passed
            return self._result(c, passed, min(max(left.confidence, right.confidence), near.confidence), "hand/tool and tool/target interaction" if passed else "interaction not established", {"hand": left.model_dump() if left.passed else right.model_dump(), "target": near.model_dump()})
        if type_ == "duration":
            seconds = float(c.get("seconds", 0)); elapsed = float(c.get("observed_seconds", 0))
            return self._result(c, elapsed >= seconds, min(1, elapsed/max(seconds, .001)), f"duration {elapsed:.2f}/{seconds:.2f}s")
        if type_ == "step_completed":
            passed = c["step_id"] in completed
            return self._result(c, passed, 1 if passed else 0, f"step {c['step_id']} {'completed' if passed else 'not completed'}")
        if type_ == "vlm_confirmation":
            result = obs.vlm_result or {}; status = result.get("step_status", "UNKNOWN")
            expected_step = c.get("step_id")
            step_matches = expected_step is None or result.get("current_step_id") in {None, expected_step}
            passed = status in {"IN_PROGRESS", "COMPLETED"} and step_matches and result.get("confidence", 0) >= c.get("min_confidence", .5)
            return self._result(c, passed, float(result.get("confidence", 0)), result.get("scene_summary", "VLM unavailable"), {"vlm": result})
        raise ValueError(f"Unsupported condition type: {type_}")

    def _near_objects(self, c: dict[str, Any], obs: Observation, left_name: str, right_name: str) -> ConditionResult:
        for left in self._objects(obs, left_name):
            for right in self._objects(obs, right_name):
                left_center = left.center(); right_center = right.center(); d = distance((left_center[0]/obs.width, left_center[1]/obs.height), (right_center[0]/obs.width, right_center[1]/obs.height))
                if d <= c["max_distance"]: return self._result(c, True, min(left.confidence, right.confidence), f"{left_name} near {right_name} ({d:.3f})", {"track_ids": [left.track_id, right.track_id], "frame_id": obs.frame_id})
        return self._result(c, False, 0, f"{left_name} not near {right_name}")
