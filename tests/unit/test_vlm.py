from app.core.config import VLMSettings
from app.vlm.openai_compatible_provider import OpenAICompatibleProvider
from app.vlm.schemas import parse_vlm_response


def test_json_parse_and_invalid_fallback():
    good=parse_vlm_response('{"scene_summary":"ok","detected_action":"x","step_status":"IN_PROGRESS","confidence":0.8,"safety_violation":false,"violations":[],"evidence":[],"uncertainties":[]}')
    assert good.step_status=="IN_PROGRESS"
    assert parse_vlm_response("not json").step_status=="UNKNOWN"


def test_json_parse_repairs_a_trailing_quote_after_an_array():
    value = '{"scene_summary":"ok","confidence":0.5,"uncertainties":[]"}'
    assert parse_vlm_response(value).scene_summary == "ok"


def test_json_parse_normalizes_string_evidence():
    value = '{"scene_summary":"ok","confidence":0.5,"evidence":"pose data"}'
    result = parse_vlm_response(value)
    assert result.evidence[0].description == "pose data"


def test_json_parse_recovers_nonstandard_local_vlm_output():
    value = '''{"scene_summary":"empty scene","detected_action":"UNKNOWN","confidence":-1,"evidence":["no pose"],"uncertainties":[No activity found]}'''
    result = parse_vlm_response(value)
    assert result.provider_success is True
    assert result.confidence == 0
    assert result.evidence[0].description == "no pose"


def test_json_parse_normalizes_local_vlm_schema_variations():
    value = {"scene_summary": "ok", "confidence": "-1", "step_status": "idle", "evidence": [{"detail": "person detected"}], "violations": "none"}
    result = parse_vlm_response(value)
    assert result.confidence == 0
    assert result.step_status == "UNKNOWN"
    assert result.evidence[0].description == "{'detail': 'person detected'}"
    assert result.violations == ["none"]


def test_openai_compatible_provider_allows_an_empty_api_key():
    settings = VLMSettings(model="local-vlm", base_url="http://127.0.0.1:8000/v1", api_key="")
    provider = OpenAICompatibleProvider(settings)
    assert provider.api_key == ""
