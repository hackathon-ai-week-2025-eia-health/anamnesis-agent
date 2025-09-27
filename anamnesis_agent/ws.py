from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, BaseMessage

from .state import AgentData, build_initial_data, clone_data

PostJsonFn = Callable[[Dict[str, Any]], None]


def extract_last_ai_message(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def public_state(data: AgentData) -> Dict[str, Any]:
    public: Dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        public[key] = clone_data(value)
    return public


async def stream_langgraph_to_ws(
    app,
    post_json: PostJsonFn,
    messages: List[BaseMessage],
    uuid_value: str,
) -> str:
    post_json({"type": "start", "uuid": uuid_value})
    data = build_initial_data(messages)
    data.setdefault("_internal", {})["uuid"] = uuid_value
    input_state: AgentState = {"messages": messages, "data": data}

    try:
        final_state = await app.ainvoke(input_state)
        final_data: AgentData = final_state.get("data", {})
        events = final_data.get("_internal", {}).get("ws_events", [])
        for event in sorted(events, key=lambda e: e.get("seq", 0)):
            payload = {
                "type": event.get("type"),
                "uuid": event.get("uuid", uuid_value),
            }
            if event.get("type") == "chunk":
                payload["data"] = event.get("data", "")
                payload["uuid"] = uuid_value
            else:
                payload["name"] = event.get("name")
                payload["status"] = event.get("status")
                if event.get("type") == "tool_result":
                    payload["output"] = event.get("output")
            post_json(payload)
        answer = extract_last_ai_message(final_state.get("messages", []))
        public = public_state(final_data)
        post_json({"type": "end", "uuid": uuid_value, "answer": answer, "state": public})
        return answer
    except Exception as exc:
        post_json({"type": "error", "uuid": uuid_value, "message": str(exc)})
        raise


__all__ = ["extract_last_ai_message", "public_state", "stream_langgraph_to_ws"]
