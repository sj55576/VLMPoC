from app.vlm.schemas import parse_vlm_response
from app.vlm.openai_compatible_provider import OpenAICompatibleProvider

def test_json_parse_and_invalid_fallback():
    good=parse_vlm_response('{"scene_summary":"ok","detected_action":"x","step_status":"IN_PROGRESS","confidence":0.8,"safety_violation":false,"violations":[],"evidence":[],"uncertainties":[]}')
    assert good.step_status=="IN_PROGRESS"
    assert parse_vlm_response("not json").step_status=="UNKNOWN"


def test_openai_compatible_provider_allows_an_empty_api_key():
    provider = OpenAICompatibleProvider("local-vlm", "http://127.0.0.1:8000/v1", "")
    assert provider.api_key == ""
