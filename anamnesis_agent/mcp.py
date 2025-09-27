import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .state import AgentData, next_event_seq, now_iso, safe_payload

ToolHandler = Callable[[Dict[str, Any]], Any]

_LAMBDA_CLIENT: Optional[Any] = None

LAMBDA_TOOL_MAP = {
    "mcp_redflags_check": "comprehend_medical.detect_entities_v2",
    "mcp_classifier": "disease_xgb.predict",
}


def _lambda_enabled(nombre_funcion: str) -> bool:
    if nombre_funcion not in LAMBDA_TOOL_MAP:
        return False
    return bool(os.getenv("ANAMNESIS_LAMBDA_FUNCTION_NAME"))


def _get_lambda_client() -> Any:
    global _LAMBDA_CLIENT
    if _LAMBDA_CLIENT is None:
        region = os.getenv("ANAMNESIS_AWS_REGION")
        client_kwargs: Dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        _LAMBDA_CLIENT = boto3.client("lambda", **client_kwargs)
    return _LAMBDA_CLIENT


def _invoke_lambda(nombre_funcion: str, payload: Dict[str, Any], necesidad: str) -> Any:
    function_name = os.getenv("ANAMNESIS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        raise RuntimeError("ANAMNESIS_LAMBDA_FUNCTION_NAME is not configured")

    client = _get_lambda_client()
    request_body: Dict[str, Any] = {
        "nombre_funcion": nombre_funcion,
        "data_enviada_json": payload,
    }
    if necesidad:
        request_body["necesidad"] = necesidad

    try:
        response = client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(request_body).encode("utf-8"),
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Lambda invocation failed: {exc}") from exc

    response_payload = response.get("Payload")
    body_bytes = response_payload.read() if response_payload else b""
    body_text = body_bytes.decode("utf-8") if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)

    try:
        envelope = json.loads(body_text) if body_text else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Lambda returned invalid JSON body: {body_text}") from exc

    status_code = envelope.get("statusCode", 200)
    if status_code >= 400:
        raise RuntimeError(f"Lambda error {status_code}: {envelope.get('body')}")

    body = envelope.get("body")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw_body": body}
    return body


def call_mcp_tool(
    data: AgentData,
    nombre_funcion: str,
    payload: Dict[str, Any],
    necesidad: str,
    handler: ToolHandler,
) -> Any:
    internal = data.setdefault("_internal", {})
    event_uuid = str(uuid.uuid4())
    seq_start = next_event_seq(internal)
    internal.setdefault("ws_events", []).append(
        {
            "type": "tool_call",
            "name": nombre_funcion,
            "status": "running",
            "uuid": event_uuid,
            "seq": seq_start,
        }
    )

    result: Any = None
    mode = "mock"
    lambda_error: Optional[str] = None

    if _lambda_enabled(nombre_funcion):
        lambda_target = LAMBDA_TOOL_MAP[nombre_funcion]
        try:
            result = _invoke_lambda(lambda_target, payload, necesidad)
            mode = "lambda"
        except Exception as exc:  # noqa: BLE001
            lambda_error = str(exc)
            result = None

    if result is None:
        result = handler(payload)
        mode = "mock" if lambda_error else mode

    seq_end = next_event_seq(internal)
    event_payload: Dict[str, Any] = {
        "type": "tool_result",
        "name": nombre_funcion,
        "status": "ok" if not lambda_error else "fallback",
        "uuid": event_uuid,
        "output": result,
        "seq": seq_end,
        "mode": mode,
    }
    if lambda_error:
        event_payload["error"] = lambda_error
    internal.setdefault("ws_events", []).append(event_payload)

    audit_entry: Dict[str, Any] = {
        "nombre_funcion": nombre_funcion,
        "data_enviada_json": safe_payload(payload),
        "necesidad": necesidad,
        "result": safe_payload(result) if isinstance(result, dict) else result,
        "mode": mode,
        "ts": now_iso(),
    }
    if lambda_error:
        audit_entry["lambda_error"] = lambda_error
    data.setdefault("tool_calls_audit", []).append(audit_entry)
    return result


# ================== MCP MOCKS ==================


def mock_redflags(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = (payload.get("texto_resumen") or "") + " " + (payload.get("motivo") or "")
    summary_lower = summary.lower()
    triggers: List[str] = []
    if "dolor en el pecho" in summary_lower and (
        "sudor frio" in summary_lower or "sudor fr\u00edo" in summary_lower
    ):
        triggers.append("Dolor toracico con sudoracion intensa")
    return {"present": bool(triggers), "triggers": triggers}


def mock_normalize(anamnesis: Dict[str, Any]) -> Dict[str, Any]:
    sympt = anamnesis.get("sintoma_principal", {})
    return {
        "chief_complaint": anamnesis.get("motivo"),
        "primary_symptom": {
            "name": sympt.get("nombre"),
            "onset": sympt.get("inicio"),
            "duration_hours": sympt.get("duracion_horas"),
            "course": sympt.get("curso"),
            "intensity": sympt.get("intensidad_0_10"),
            "relieving_factors": sympt.get("factores", {}).get("alivia", []),
            "aggravating_factors": sympt.get("factores", {}).get("agrava", []),
        },
        "medical_history": {
            "personal": anamnesis.get("antecedentes_personales", []),
            "family": anamnesis.get("antecedentes_familiares", []),
        },
        "lifestyle": {
            "risk_habits": anamnesis.get("habitos_riesgo", []),
        },
        "associated_symptoms": anamnesis.get("sintomas_asociados", []),
    }


def mock_translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = payload.get("structured") or {}
    return {
        "lang": "en",
        "structured": structured,
        "notice": "Mock translation applied",
    }


def mock_classifier(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"condicion": "sindrome coronario agudo", "prob": 0.72},
        {"condicion": "angina estable", "prob": 0.18},
        {"condicion": "ansiedad", "prob": 0.10},
    ]


def mock_calibrate(predicciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = sum(p.get("prob", 0) for p in predicciones) or 1.0
    calibrated: List[Dict[str, Any]] = []
    for pred in predicciones:
        prob = pred.get("prob", 0) / total
        calibrated.append({**pred, "prob": round(prob, 4)})
    return calibrated


def mock_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
    predictions = payload.get("predicciones") or []
    top_prob = predictions[0].get("prob", 0) if predictions else 0
    if top_prob >= 0.6:
        nivel = "urgencias"
        razon = [
            "Probabilidad alta de condicion tiempo-dependiente.",
            "Acude a emergencias de inmediato.",
        ]
    elif top_prob >= 0.3:
        nivel = "consultar"
        razon = [
            "Requiere valoracion medica presencial en las proximas 24 horas.",
            "Busca un servicio de guardia o consulta prioritaria.",
        ]
    else:
        nivel = "autocuidado"
        razon = [
            "Puedes realizar autocuidado con vigilancia estrecha.",
            "Si empeora, consulta a un profesional.",
        ]
    return {"nivel": nivel, "razon": razon}


def mock_health_links(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    lang = payload.get("lang", "es")
    if lang == "es":
        return [
            {
                "title": "OMS - Reconocer signos de alarma",
                "url": "https://www.who.int/es/",
            },
            {
                "title": "Ministerio de Salud - Orientacion telefonica",
                "url": "https://www.argentina.gob.ar/salud",
            },
        ]
    return [
        {
            "title": "WHO - Chest Pain Guidance",
            "url": "https://www.who.int/",
        }
    ]


__all__ = [
    "call_mcp_tool",
    "mock_redflags",
    "mock_normalize",
    "mock_translate",
    "mock_classifier",
    "mock_calibrate",
    "mock_triage",
    "mock_health_links",
]
