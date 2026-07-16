from app.vlm.schemas import parse_vlm_response

def test_json_parse_and_invalid_fallback():
    good=parse_vlm_response('{"scene_summary":"ok","detected_action":"x","step_status":"IN_PROGRESS","confidence":0.8,"safety_violation":false,"violations":[],"evidence":[],"uncertainties":[]}')
    assert good.step_status=="IN_PROGRESS"
    assert parse_vlm_response("not json").step_status=="UNKNOWN"
