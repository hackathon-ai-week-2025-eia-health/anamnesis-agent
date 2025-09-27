# file: lambda_ws_langgraph_bedrock.py
import os
import json
import asyncio
import base64
from typing import List, Dict, Any

import boto3
from botocore.config import Config
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_aws import ChatBedrock

# ============ Config ============
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.deepseek.r1-v1:0")

_boto_cfg = Config(
    region_name=AWS_REGION, retries={"max_attempts": 3, "mode": "standard"}
)


# ============ LangGraph ============
def get_bedrock_chat():
    return ChatBedrock(
        model=BEDROCK_MODEL_ID,
        streaming=True,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )


State = MessagesState


def llm_node(state: State):
    llm = get_bedrock_chat()
    ai_msg = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)
    return graph.compile()


APP = build_graph()


# ============ WebSocket utils ============
def _ws_client(event) -> boto3.client:
    domain = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    endpoint = f"https://{domain}/{stage}"
    return boto3.client(
        "apigatewaymanagementapi", endpoint_url=endpoint, config=_boto_cfg
    )


def _post_json(ws, connection_id: str, payload: Dict[str, Any]):
    ws.post_to_connection(
        ConnectionId=connection_id,
        Data=json.dumps(payload).encode("utf-8"),
    )


async def _stream_langgraph_to_ws(
    app, ws, connection_id: str, messages: List[BaseMessage], uuid: str
):
    _post_json(ws, connection_id, {"type": "start", "uuid": uuid})
    _input = {"messages": messages}

    try:
        async for ev in app.astream_events(_input, version="v1"):
            if ev.get("event") == "on_chat_model_stream":
                chunk = ev["data"]["chunk"]
                if chunk and getattr(chunk, "content", None):
                    _post_json(
                        ws,
                        connection_id,
                        {"type": "chunk", "uuid": uuid, "data": chunk.content},
                    )

        final_state = app.invoke(_input)
        last_ai = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg.content
                break

        _post_json(ws, connection_id, {"type": "end", "uuid": uuid, "answer": last_ai})
        return last_ai
    except Exception as e:
        _post_json(
            ws, connection_id, {"type": "error", "uuid": uuid, "message": str(e)}
        )
        raise


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


# ============ Lambda Handler ============
def lambda_handler(event, context):
    route = event.get("requestContext", {}).get("routeKey", "$default")
    # --- MODO PRUEBA EN CONSOLA (sin WebSocket, sin streaming) ---
    if route == "CONSOLE_TEST":
        # El Event JSON de la consola incluirá 'messages' directamente
        raw_messages = event.get("messages", [])
        messages = _parse_messages(raw_messages)
        final_state = APP.invoke({"messages": messages})
        last_ai = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg.content
                break
        return {
            "statusCode": 200,
            "body": json.dumps({"answer": last_ai})
        }
    # -------------------------------------------------------------
    
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

        action = payload.get("action", "sendMessage")
        uuid = payload.get("uuid")
        raw_messages = payload.get("messages", [])

        if not uuid or not isinstance(raw_messages, list):
            _post_json(
                ws,
                connection_id,
                {
                    "type": "error",
                    "uuid": uuid or "",
                    "message": "Invalid payload. Expecting { uuid, messages[list] }.",
                },
            )
            return {"statusCode": 400, "body": "bad request"}

        messages = _parse_messages(raw_messages)

        asyncio.run(_stream_langgraph_to_ws(APP, ws, connection_id, messages, uuid))
        return {"statusCode": 200, "body": "streamed"}
    except Exception as e:
        try:
            if connection_id:
                _post_json(
                    ws, connection_id, {"type": "error", "uuid": "", "message": str(e)}
                )
        except Exception:
            pass
        return {"statusCode": 500, "body": "error"}