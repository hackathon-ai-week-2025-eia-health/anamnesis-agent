"""Local test client for the anamnesis LangGraph agent."""
import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from anamnesis_agent.graph import build_graph
from anamnesis_agent.ws import stream_langgraph_to_ws

DEFAULT_SCENARIOS: Dict[str, List[Dict[str, str]]] = {
    "normal": [
        {
            "role": "user",
            "content": "Hola, tengo un dolor en el pecho desde hace unas horas y quiero saber si debo preocuparme.",
        },
        {
            "role": "assistant",
            "content": "Para avanzar necesito saber el motivo principal de tu consulta. ¿Podrías contármelo, por favor?",
        },
        {
            "role": "user",
            "content": "El motivo es que estoy preocupado por este dolor y quiero saber qué puede ser.",
        },
        {
            "role": "assistant",
            "content": "Para avanzar necesito saber cuál es el síntoma principal. Por ejemplo: dolor en el pecho, tos persistente. ¿Podrías contármelo, por favor?",
        },
        {
            "role": "user",
            "content": "El síntoma principal es un dolor opresivo en el pecho.",
        },
    ],
    "red_flag": [
        {
            "role": "user",
            "content": "Tengo un dolor muy fuerte en el pecho y un sudor frío, ¿qué hago?",
        }
    ],
    "stop": [
        {
            "role": "user",
            "content": "Empecemos con la anamnesis",
        },
        {
            "role": "assistant",
            "content": "Para avanzar necesito saber el motivo principal de tu consulta. ¿Podrías contármelo, por favor?",
        },
        {
            "role": "user",
            "content": "No quiero continuar",
        },
    ],
}


def _parse_messages(raw_turns: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert role/content dicts into LangChain messages."""
    messages: List[BaseMessage] = []
    for turn in raw_turns:
        role = (turn.get("role") or "").lower()
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            raise ValueError(f"Unsupported role '{role}' in turn: {turn}")
    return messages


def _load_messages(args: argparse.Namespace) -> List[BaseMessage]:
    if args.messages_file:
        path = Path(args.messages_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        return _parse_messages(data)
    if args.messages_json:
        try:
            data = json.loads(args.messages_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in --messages-json: {exc}") from exc
        return _parse_messages(data)
    scenario = args.scenario or "normal"
    if scenario not in DEFAULT_SCENARIOS:
        raise SystemExit(f"Unknown scenario '{scenario}'. Available: {', '.join(DEFAULT_SCENARIOS)}")
    return _parse_messages(DEFAULT_SCENARIOS[scenario])


def _make_event_printer(verbose: bool):
    def _printer(payload: Dict[str, Any]):
        if not verbose and payload.get("type") not in {"chunk", "end", "error"}:
            return
        print(json.dumps(payload, ensure_ascii=False))

    return _printer


async def _run_client(args: argparse.Namespace) -> None:
    classifier_lang = args.classifier_lang or os.getenv("CLASSIFIER_TARGET_LANG", "es")
    app = build_graph(classifier_lang)
    messages = _load_messages(args)
    event_uuid = args.uuid or str(uuid.uuid4())
    printer = _make_event_printer(args.verbose)
    await stream_langgraph_to_ws(app, printer, messages, event_uuid)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a local conversation against the anamnesis LangGraph agent.")
    parser.add_argument("--scenario", choices=sorted(DEFAULT_SCENARIOS.keys()), help="Built-in scenario to run")
    parser.add_argument("--messages-file", help="Path to JSON file with conversation turns [{role,content}]")
    parser.add_argument("--messages-json", help="Inline JSON string with conversation turns")
    parser.add_argument("--classifier-lang", help="Override classifier target language (default uses env or 'es')")
    parser.add_argument("--uuid", help="Conversation UUID (auto-generated if omitted)")
    parser.add_argument("--verbose", action="store_true", help="Print all tool events, not just chunks/end/error")

    args = parser.parse_args(argv)

    try:
        asyncio.run(_run_client(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - convenience for manual runs
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
