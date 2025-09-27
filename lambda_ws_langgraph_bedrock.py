# file: lambda_ws_langgraph_bedrock.py
import os
import json
import asyncio
import base64
from typing import Dict, Any, List

import boto3
from botocore.config import Config
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from anamnesis_agent.graph import build_graph
from anamnesis_agent.state import build_initial_data
from anamnesis_agent.ws import (
    extract_last_ai_message,
    public_state,
    stream_langgraph_to_ws,
)

# ============ Config ============
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-west-2")
CLASSIFIER_TARGET_LANG = os.getenv("CLASSIFIER_TARGET_LANG", "es").strip().lower()

_boto_cfg = Config(
    region_name=AWS_REGION, retries={"max_attempts": 3, "mode": "standard"}
)

# Configuración del grafo con límite de recursión más alto
APP = build_graph(CLASSIFIER_TARGET_LANG)


# ============ WebSocket utils ============
def _ws_client(event) -> boto3.client:
    domain = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    endpoint = f"https://{domain}/{stage}"
    return boto3.client(
        "apigatewaymanagementapi", endpoint_url=endpoint, config=_boto_cfg
    )


def _make_post_json(ws, connection_id: str):
    def _post(payload: Dict[str, Any]):
        ws.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(payload).encode("utf-8"),
        )

    return _post


def _parse_messages(raw: List[Dict[str, str]]) -> List[BaseMessage]:
    chat: List[BaseMessage] = []
    for m in raw or []:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if role == "user":
            chat.append(HumanMessage(content=content))
        else:
            chat.append(AIMessage(content=content))
    return chat


async def _run_stream(
    app, ws, connection_id: str, messages: List[BaseMessage], uuid_value: str
):
    post_json = _make_post_json(ws, connection_id)
    await stream_langgraph_to_ws(app, post_json, messages, uuid_value)


# ============ Lambda Handler ============
def lambda_handler(event, context):
    route = event.get("requestContext", {}).get("routeKey", "$default")
    if route == "CONSOLE_TEST":
        raw_messages = event.get("messages", [])
        messages = _parse_messages(raw_messages)
        data = build_initial_data(messages)
        data.setdefault("_internal", {})["uuid"] = "console"
        final_state = APP.invoke({"messages": messages, "data": data})
        answer = extract_last_ai_message(final_state.get("messages", []))
        state_public = public_state(final_state.get("data", {}))
        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer, "state": state_public}),
        }

    connection_id = event.get("requestContext", {}).get("connectionId")
    ws = _ws_client(event)

    if route in ("$connect", "$disconnect"):
        return {"statusCode": 200, "body": "ok"}

    if route != "$default":
        return {"statusCode": 200, "body": "ok"}

    try:
        body_raw = event.get("body", "")
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw or "").decode("utf-8")
        payload = json.loads(body_raw or "{}")

        uuid_value = payload.get("uuid")
        raw_messages = payload.get("messages", [])

        if not uuid_value or not isinstance(raw_messages, list):
            _make_post_json(ws, connection_id)(
                {
                    "type": "error",
                    "uuid": uuid_value or "",
                    "message": "Invalid payload. Expecting { uuid, messages[list] }.",
                }
            )
            return {"statusCode": 400, "body": "bad request"}

        messages = _parse_messages(raw_messages)
        asyncio.run(_run_stream(APP, ws, connection_id, messages, uuid_value))
        return {"statusCode": 200, "body": "streamed"}
    except Exception as e:
        try:
            if connection_id:
                _make_post_json(ws, connection_id)(
                    {"type": "error", "uuid": "", "message": str(e)}
                )
        except Exception:
            pass
        return {"statusCode": 500, "body": "error"}
