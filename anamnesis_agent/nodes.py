from typing import Any, Dict, List, Set

from langchain_core.messages import AIMessage, HumanMessage

from .constants import FIELD_CONFIG, STOP_PHRASES
from .llm import generate_final_message, generate_question, llm_enabled
from .mcp import (
    call_mcp_tool,
    mock_calibrate,
    mock_classifier,
    mock_health_links,
    mock_normalize,
    mock_redflags,
    mock_translate,
    mock_triage,
)
from .state import (
    AgentState,
    clone_data,
    default_anamnesis,
    detect_language_from_dialog,
    determine_missing_fields,
    ensure_parsed,
    has_sufficient_data,
    record_chunk_event,
)


CLASSIFIER_TARGET_LANG = "es"


def _is_negated_entity(entity: Dict[str, Any]) -> bool:
    for attr in entity.get("attributes", []):
        if str(attr.get("Type", "")).lower() == "negation":
            return True
    for trait in entity.get("traits", []):
        key = trait.get("Name") or trait.get("Type")
        if str(key).lower() == "negation":
            return True
    return False


def _collect_entity_tokens(entities: List[Dict[str, Any]]) -> Set[str]:
    tokens: Set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if _is_negated_entity(entity):
            continue
        text = str(entity.get("text") or entity.get("Text") or "").strip().lower()
        if text:
            tokens.add(text)
    return tokens


def _infer_redflag_triggers(entities: List[Dict[str, Any]]) -> List[str]:
    tokens = _collect_entity_tokens(entities)
    triggers: List[str] = []
    chest_pain = any("chest pain" in token for token in tokens)
    dyspnea = any(
        token in {"dyspnea", "shortness of breath", "breathlessness"}
        for token in tokens
    )
    sweating = any(token in {"diaphoresis", "sweating"} for token in tokens)
    syncope = any(token in {"syncope", "loss of consciousness"} for token in tokens)
    stroke = any(token in {"stroke", "cerebrovascular accident"} for token in tokens)
    if chest_pain and dyspnea:
        triggers.append("Chest pain with respiratory compromise")
    if chest_pain and sweating:
        triggers.append("Chest pain with autonomic symptoms")
    if syncope:
        triggers.append("Syncope reported")
    if stroke:
        triggers.append("Possible acute neurologic deficit")
    return triggers


def _redflags_from_lambda(result: Dict[str, Any]) -> Dict[str, Any]:
    entities = result.get("entities") or []
    triggers = _infer_redflag_triggers(entities)
    return {
        "present": bool(triggers),
        "triggers": triggers,
        "entities": entities,
        "request_id": result.get("request_id"),
        "input_chars": result.get("input_chars"),
    }


def set_classifier_target_lang(value: str) -> None:
    global CLASSIFIER_TARGET_LANG
    CLASSIFIER_TARGET_LANG = value.strip().lower() or "es"


def detect_language_and_greet_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    raw_dialog = data.get("raw_dialog", [])
    detected_lang = detect_language_from_dialog(raw_dialog, data.get("lang", "es"))
    data["lang"] = detected_lang
    has_ai = any(entry.get("role") == "assistant" for entry in raw_dialog)
    if not has_ai:
        greeting = (
            "Hola, soy tu asistente digital para recopilar información clínica. "
            "Esto no reemplaza la valoración médica profesional, pero puedo orientarte."
        )
        record_chunk_event(data, greeting)
        raw_dialog.append({"role": "assistant", "content": greeting})
        data["raw_dialog"] = raw_dialog
        return {"data": data, "messages": [AIMessage(content=greeting)]}
    return {"data": data}


def check_consent_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    user_messages = [
        m for m in state.get("messages", []) if isinstance(m, HumanMessage)
    ]
    last_user = user_messages[-1].content.lower() if user_messages else ""
    if any(phrase in last_user for phrase in STOP_PHRASES):
        data["consent"] = "rejected"
        data["stop_reason"] = "user_declined"
    return {"data": data}


def red_flags_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    parsed = ensure_parsed(data)
    anam = parsed.get("anamnesis", {})
    payload = {
        "motivo": anam.get("motivo"),
        "sintoma": anam.get("sintoma_principal", {}).get("nombre"),
        "texto_resumen": parsed.get("latest_user_text", ""),
    }
    result = call_mcp_tool(
        data,
        "mcp_redflags_check",
        payload,
        "Descartar urgencia potencial antes de continuar.",
        mock_redflags,
    )
    processed = result
    if isinstance(result, dict) and "entities" in result:
        processed = _redflags_from_lambda(result)
    data["red_flags"] = processed
    if isinstance(processed, dict) and processed.get("present"):
        data["stop_reason"] = "red_flags"
    return {"data": data}


def planner_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})

    # Track total graph steps for emergency exit
    graph_steps = internal.get("graph_steps", 0) + 1
    internal["graph_steps"] = graph_steps

    # EMERGENCY EXIT CONDITIONS - multiple safeguards
    loop_count = internal.get("planner_loop_count", 0)
    conversation_turns = len(data.get("raw_dialog", [])) // 2

    # Immediate exit conditions (very aggressive)
    if (
        graph_steps > 20  # Too many total steps
        or loop_count > 5  # Too many planner loops
        or conversation_turns > 12  # Too long conversation
    ):
        internal["planner_action"] = "classify"
        internal["planner_target"] = None
        internal["emergency_exit"] = True
        internal["emergency_reason"] = (
            f"steps:{graph_steps}, loops:{loop_count}, turns:{conversation_turns}"
        )
        return {"data": data}

    # Continue with existing logic
    parsed = ensure_parsed(data)
    missing = determine_missing_fields(parsed)
    internal["missing_fields"] = missing

    internal["planner_loop_count"] = loop_count + 1

    # Check conversation quality - if user is giving non-responsive answers
    if _detect_conversation_stuck(dict(data), internal):
        internal["planner_action"] = "classify"
        internal["planner_target"] = None
        internal["planner_loop_count"] = 0
        return {"data": data}

    # If no missing fields, proceed to classification
    if not missing:
        internal["planner_action"] = "classify"
        internal["planner_target"] = None
        internal["planner_loop_count"] = 0
        return {"data": data}

    # Check if we're waiting for user response
    awaiting = internal.get("awaiting_field")
    if awaiting:
        # Check if user has been unresponsive for too long
        if _check_user_responsiveness(dict(data), awaiting):
            internal["planner_action"] = "await"
            internal["planner_target"] = awaiting
            return {"data": data}
        else:
            # User seems unresponsive to this field, try something else
            internal.pop("awaiting_field", None)
            recently_asked = internal.setdefault("recently_asked_fields", [])
            if awaiting not in recently_asked:
                recently_asked.append(awaiting)

    # Smart field selection with context awareness
    recently_asked = internal.get("recently_asked_fields", [])
    failed_fields = internal.get("failed_fields", [])  # Fields user couldn't answer
    available_missing = [
        f for f in missing if f not in recently_asked and f not in failed_fields
    ]

    if not available_missing:
        # If we've exhausted options, use adaptive strategy
        if _can_proceed_with_minimal_info(parsed):
            internal["planner_action"] = "classify"
            internal["planner_target"] = None
            internal["planner_loop_count"] = 0
            internal["recently_asked_fields"] = []  # Reset
            return {"data": data}
        else:
            # Try one more critical field or give up
            critical_missing = [f for f in missing if f in ["motivo", "sintoma_nombre"]]
            if critical_missing:
                target = critical_missing[0]
                internal["planner_action"] = "ask"
                internal["planner_target"] = target
                internal["awaiting_field"] = target
                return {"data": data}
            else:
                # Force classification with what we have
                internal["planner_action"] = "classify"
                internal["planner_target"] = None
                return {"data": data}

    # Intelligent field selection with conversation context
    target = _select_next_field_enhanced(
        available_missing, parsed, internal, dict(data)
    )
    internal["planner_action"] = "ask"
    internal["planner_target"] = target
    internal["awaiting_field"] = target

    # Track asked fields with smarter management
    recently_asked = internal.setdefault("recently_asked_fields", [])
    if target not in recently_asked:
        recently_asked.append(target)

    # Dynamic list management based on conversation progress
    max_recent = 3 if conversation_turns < 8 else 5
    if len(recently_asked) > max_recent:
        recently_asked = recently_asked[-max_recent:]
        internal["recently_asked_fields"] = recently_asked

    return {"data": data}


def _select_next_field_enhanced(
    missing: List[str],
    parsed: Dict[str, Any],
    internal: Dict[str, Any],
    data: Dict[str, Any],
) -> str:
    """Enhanced intelligent field selection with conversation context awareness"""
    anamnesis = parsed.get("anamnesis", {})
    conversation_turns = len(data.get("raw_dialog", [])) // 2
    failed_attempts = internal.get("failed_field_attempts", {})

    # Priority 1: Absolute critical fields - always ask first
    critical_fields = ["motivo", "sintoma_nombre"]
    for field in critical_fields:
        if field in missing:
            return field

    # Priority 2: Context-aware temporal information
    # If we have basic info, focus on temporal aspects smartly
    if anamnesis.get("motivo") and anamnesis.get("sintoma_principal", {}).get("nombre"):
        temporal_fields = ["sintoma_duracion_horas", "sintoma_inicio"]

        # Choose based on what's easier to answer
        for field in temporal_fields:
            if field in missing and failed_attempts.get(field, 0) < 2:
                # Prefer duration over inicio if user seems to struggle with time references
                if field == "sintoma_duracion_horas" and "sintoma_inicio" in missing:
                    recent_responses = _get_recent_user_responses(data, 3)
                    if any(
                        "no sé" in resp.lower() or "no recuerdo" in resp.lower()
                        for resp in recent_responses
                    ):
                        # User struggles with memory, prefer duration
                        return field
                return field

    # Priority 3: Symptom characterization - but be smart about it
    characterization_fields = ["sintoma_intensidad", "sintoma_curso"]
    for field in characterization_fields:
        if field in missing and failed_attempts.get(field, 0) < 2:
            # Intensity is usually easier than course
            if field == "sintoma_intensidad":
                return field
            elif field == "sintoma_curso" and "sintoma_intensidad" not in missing:
                return field

    # Priority 4: Medical context - adapt based on conversation flow
    if conversation_turns < 6:  # Early in conversation
        priority_context = ["antecedentes_personales", "sintomas_asociados"]
    else:  # Later in conversation
        priority_context = [
            "sintomas_asociados",
            "antecedentes_personales",
            "habitos_riesgo",
        ]

    for field in priority_context:
        if field in missing and failed_attempts.get(field, 0) < 2:
            return field

    # Priority 5: Fallback with failure consideration
    # Try fields that haven't failed too many times
    for field in missing:
        if failed_attempts.get(field, 0) < 3:
            return field

    # Last resort: return any missing field
    return missing[0] if missing else "motivo"


def _detect_conversation_stuck(data: Dict[str, Any], internal: Dict[str, Any]) -> bool:
    """Detect if conversation is stuck in unproductive patterns"""
    raw_dialog = data.get("raw_dialog", [])
    if len(raw_dialog) < 6:  # Too early to detect patterns
        return False

    # Check for repetitive assistant questions
    recent_assistant_msgs = [
        turn.get("content", "").lower()
        for turn in raw_dialog[-6:]
        if turn.get("role") == "assistant"
    ]

    if len(recent_assistant_msgs) >= 3:
        # Check for similar questions
        for i in range(len(recent_assistant_msgs) - 1):
            for j in range(i + 1, len(recent_assistant_msgs)):
                if _questions_too_similar(
                    recent_assistant_msgs[i], recent_assistant_msgs[j]
                ):
                    return True

    # Check for user frustration patterns
    recent_user_msgs = [
        turn.get("content", "").lower()
        for turn in raw_dialog[-4:]
        if turn.get("role") == "user"
    ]

    frustration_indicators = [
        "ya te dije",
        "ya te conté",
        "no sé qué más",
        "no entiendo",
        "no sé",
        "no recuerdo",
        "no lo sé",
        "basta",
        "termina",
    ]

    frustration_count = sum(
        1
        for msg in recent_user_msgs
        for indicator in frustration_indicators
        if indicator in msg
    )

    return frustration_count >= 2


def _questions_too_similar(q1: str, q2: str) -> bool:
    """Check if two questions are too similar (indicate repetitive behavior)"""
    # Simple similarity check based on key terms
    key_terms_q1 = set(word for word in q1.split() if len(word) > 3)
    key_terms_q2 = set(word for word in q2.split() if len(word) > 3)

    if not key_terms_q1 or not key_terms_q2:
        return False

    overlap = len(key_terms_q1.intersection(key_terms_q2))
    similarity = overlap / min(len(key_terms_q1), len(key_terms_q2))

    return similarity > 0.6


def _check_user_responsiveness(data: Dict[str, Any], awaiting_field: str) -> bool:
    """Check if user is being responsive to the awaiting field"""
    raw_dialog = data.get("raw_dialog", [])
    if len(raw_dialog) < 2:
        return True  # Give benefit of doubt early in conversation

    # Look for the last assistant question about this field
    field_config = FIELD_CONFIG.get(awaiting_field, {})
    field_keywords = field_config.get("keywords", [])

    # Count recent attempts to ask about this field
    recent_attempts = 0
    for i in range(len(raw_dialog) - 1, -1, -1):
        turn = raw_dialog[i]
        if turn.get("role") == "assistant":
            content = turn.get("content", "").lower()
            if any(keyword in content for keyword in field_keywords):
                recent_attempts += 1
                if recent_attempts >= 2:  # Asked twice recently
                    return False  # User not responsive
        elif recent_attempts > 0:
            # Found user response, check if it was relevant
            user_content = turn.get("content", "").lower()
            if any(
                indicator in user_content
                for indicator in ["no sé", "no recuerdo", "no lo sé"]
            ):
                return False  # User can't/won't answer
            break

    return True


def _can_proceed_with_minimal_info(parsed: Dict[str, Any]) -> bool:
    """Enhanced check for minimal viable information"""
    anamnesis = parsed.get("anamnesis", {})

    # Must have basic complaint
    if not anamnesis.get("motivo") and not anamnesis.get("sintoma_principal", {}).get(
        "nombre"
    ):
        return False

    # Must have SOME temporal or descriptive information
    sympt = anamnesis.get("sintoma_principal", {})
    has_temporal = bool(sympt.get("inicio") or sympt.get("duracion_horas"))
    has_descriptive = bool(sympt.get("intensidad_0_10") or sympt.get("curso"))
    has_context = bool(
        anamnesis.get("antecedentes_personales") or anamnesis.get("sintomas_asociados")
    )

    # Need at least 2 out of 3 categories
    info_categories = sum([has_temporal, has_descriptive, has_context])

    return info_categories >= 2


def _get_recent_user_responses(data: Dict[str, Any], count: int = 3) -> List[str]:
    """Get recent user responses for analysis"""
    raw_dialog = data.get("raw_dialog", [])
    user_responses = [
        turn.get("content", "") for turn in raw_dialog if turn.get("role") == "user"
    ]
    return user_responses[-count:] if user_responses else []


def ask_user_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})

    # Track graph steps
    graph_steps = internal.get("graph_steps", 0) + 1
    internal["graph_steps"] = graph_steps

    # Emergency exit if too many steps
    if graph_steps > 21:
        # Don't ask more questions, force classification
        internal["emergency_exit"] = True
        return {"data": data}

    if internal.get("planner_action") != "ask":
        return {"data": data}

    target = internal.get("planner_target")
    if not target or not isinstance(target, str):
        return {"data": data}

    config = FIELD_CONFIG.get(target)
    if not config:
        return {"data": data}

    # Enhanced question generation with context awareness
    internal["last_asked_field"] = target

    # Check if this is a retry (field has been asked before)
    failed_attempts = internal.get("failed_field_attempts", {})
    is_retry = failed_attempts.get(target, 0) > 0
    conversation_turns = len(data.get("raw_dialog", [])) // 2

    question = None

    # Try intelligent question generation with LLM
    if llm_enabled():
        question = generate_question(
            target,
            config,
            data.get("anamnesis", default_anamnesis()),
            data.get("lang", "es"),
        )
        if question:
            question = question.strip()

    # Enhanced fallback questions with context awareness
    if not question:
        question = _generate_contextual_question(
            target, config, dict(data), is_retry, conversation_turns
        )

    # Add encouraging context for retries
    if is_retry and question:
        encouragement = _get_encouragement_prefix(target, data.get("lang", "es"))
        if encouragement:
            question = f"{encouragement} {question}"

    record_chunk_event(data, question)
    raw_dialog = data.get("raw_dialog", [])
    raw_dialog.append({"role": "assistant", "content": question})
    data["raw_dialog"] = raw_dialog
    return {"data": data, "messages": [AIMessage(content=question)]}


def _generate_contextual_question(
    target: str,
    config: Dict[str, Any],
    data: Dict[str, Any],
    is_retry: bool,
    conversation_turns: int,
) -> str:
    """Generate contextual questions based on conversation state and field type"""

    anamnesis = data.get("anamnesis", {})

    # Get what we already know for context
    known_symptom = anamnesis.get("sintoma_principal", {}).get("nombre")
    known_motivo = anamnesis.get("motivo")

    # Context-aware question templates
    contextual_questions = {
        "motivo": [
            "¿Qué te trae por aquí hoy?"
            if not is_retry
            else "¿Podrías contarme brevemente cuál es tu principal preocupación?",
            "¿Cuál es el motivo principal de tu consulta?"
            if conversation_turns > 3
            else "¿En qué puedo ayudarte hoy?",
        ],
        "sintoma_nombre": [
            f"Veo que mencionaste '{known_motivo}', ¿podrías describirme específicamente qué síntoma tienes?"
            if known_motivo
            else "¿Podrías describirme el síntoma principal que estás experimentando?",
            "¿Qué es exactamente lo que sientes?"
            if is_retry
            else "¿Cuál es el síntoma que más te preocupa?",
        ],
        "sintoma_inicio": [
            f"¿Cuándo comenzó {known_symptom or 'este síntoma'}?"
            if known_symptom
            else "¿Cuándo empezaste a notar este problema?",
            "¿Desde cuándo tienes esta molestia?"
            if is_retry
            else "¿Recuerdas cuándo empezó?",
        ],
        "sintoma_duracion_horas": [
            f"¿Cuánto tiempo llevas con {known_symptom or 'este síntoma'}?"
            if known_symptom
            else "¿Cuánto tiempo ha estado presente?",
            "¿Podrías decirme si han sido horas, días o más tiempo?"
            if is_retry
            else "¿Ha sido continuo todo este tiempo?",
        ],
        "sintoma_curso": [
            f"¿{known_symptom or 'El síntoma'} es constante o va y viene?"
            if known_symptom
            else "¿Este problema es constante o intermitente?",
            "¿Lo sientes todo el tiempo o solo a ratos?"
            if is_retry
            else "¿Cómo se comporta a lo largo del tiempo?",
        ],
        "sintoma_intensidad": [
            f"En una escala del 0 al 10, ¿qué tan intenso es {known_symptom or 'el síntoma'}?"
            if known_symptom
            else "¿Qué tan fuerte es, del 0 al 10?",
            "¿Podrías darme un número del 0 al 10 para la intensidad?"
            if is_retry
            else "¿Lo consideras leve, moderado o fuerte?",
        ],
        "antecedentes_personales": [
            "¿Tienes algún problema de salud previo o tomas algún medicamento?"
            if conversation_turns < 5
            else "¿Hay algo en tu historial médico que debería saber?",
            "¿Algún antecedente médico importante?"
            if is_retry
            else "¿Tomas medicamentos o has tenido problemas de salud antes?",
        ],
        "antecedentes_familiares": [
            "¿Hay problemas de salud importantes en tu familia cercana?",
            "¿Algún familiar ha tenido algo similar?"
            if is_retry
            else "¿Tu familia tiene historial de alguna enfermedad relevante?",
        ],
        "habitos_riesgo": [
            "¿Tienes algún hábito como fumar, beber alcohol o alguna exposición laboral?",
            "¿Fumas, bebes o tienes algún hábito que consideres relevante?"
            if is_retry
            else "¿Hay algo en tu estilo de vida que podría estar relacionado?",
        ],
        "sintomas_asociados": [
            f"¿Has notado otros síntomas además de {known_symptom or 'este'}?"
            if known_symptom
            else "¿Tienes algún otro síntoma que acompañe a este problema?",
            "¿Algo más que hayas notado?"
            if is_retry
            else "¿Hay otros síntomas que vengan junto con esto?",
        ],
    }

    questions = contextual_questions.get(target, [])
    if questions:
        # Choose first question unless it's a retry, then use second if available
        question_index = 1 if is_retry and len(questions) > 1 else 0
        return questions[question_index]

    # Fallback to basic question
    example = config.get("example", "")
    basic_question = f"Para continuar, necesito saber {config['description']}."
    if example and not is_retry:
        basic_question += f" Por ejemplo: {example}."
    basic_question += " ¿Podrías contármelo?"

    return basic_question


def _get_encouragement_prefix(target: str, lang: str) -> str:
    """Get encouraging prefix for retry questions"""
    if lang == "es":
        encouragements = {
            "sintoma_duracion_horas": "Entiendo que puede ser difícil recordar exactamente.",
            "sintoma_inicio": "No te preocupes si no recuerdas exactamente.",
            "antecedentes_personales": "Cualquier información que puedas compartir es útil.",
            "antecedentes_familiares": "Solo lo que recuerdes está bien.",
            "habitos_riesgo": "Comparte solo lo que consideres relevante.",
        }
        return encouragements.get(target, "")
    return ""


def parse_and_clean_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    ensure_parsed(data)
    return {"data": data}


def update_state_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})

    # Track graph steps for emergency detection
    graph_steps = internal.get("graph_steps", 0) + 1
    internal["graph_steps"] = graph_steps

    # Emergency check - if we're in too deep, force classification
    if graph_steps > 22:
        internal["emergency_exit"] = True
        internal["has_sufficiency"] = True
        return {"data": data}

    parsed = ensure_parsed(data)
    internal = data.setdefault("_internal", {})

    # Update anamnesis
    old_anamnesis = data.get("anamnesis", {})
    new_anamnesis = parsed.get("anamnesis", default_anamnesis())
    data["anamnesis"] = new_anamnesis

    # Enhanced information tracking
    info_added = _compare_anamnesis(old_anamnesis, new_anamnesis)

    # Track field success/failure rates
    awaiting_field = internal.get("awaiting_field")
    if awaiting_field:
        failed_attempts = internal.setdefault("failed_field_attempts", {})

        if info_added:
            # Success - field was answered
            internal.pop("awaiting_field", None)
            # Reset failure count for this field
            failed_attempts.pop(awaiting_field, None)
            # Reset loop count on successful progress
            internal["planner_loop_count"] = 0
            internal.pop("planner_action", None)
        else:
            # No new info - potential failure
            recent_user_response = _get_last_user_response(dict(data))
            if _is_evasive_response(recent_user_response, awaiting_field):
                # Mark this field as failed
                failed_attempts[awaiting_field] = (
                    failed_attempts.get(awaiting_field, 0) + 1
                )
                # If failed too many times, add to failed fields list
                if failed_attempts[awaiting_field] >= 2:
                    failed_fields = internal.setdefault("failed_fields", [])
                    if awaiting_field not in failed_fields:
                        failed_fields.append(awaiting_field)

                # Clear awaiting field to try something else
                internal.pop("awaiting_field", None)
                internal.pop("planner_action", None)
    else:
        # No specific field was being awaited, but check if we got useful info anyway
        if info_added:
            internal["planner_loop_count"] = 0
            internal.pop("planner_action", None)

    return {"data": data}


def _get_last_user_response(data: Dict[str, Any]) -> str:
    """Get the most recent user response"""
    raw_dialog = data.get("raw_dialog", [])
    for turn in reversed(raw_dialog):
        if turn.get("role") == "user":
            return turn.get("content", "").strip()
    return ""


def _is_evasive_response(response: str, expected_field: str) -> bool:
    """Enhanced check for evasive or non-responsive answers"""
    if not response:
        return True

    response_lower = response.lower().strip()

    # Direct evasive responses
    evasive_patterns = [
        "no sé",
        "no lo sé",
        "no estoy seguro",
        "no recuerdo",
        "no me acuerdo",
        "no sabría decir",
        "no tengo idea",
        "no lo recuerdo",
        "ya te dije",
        "ya te conté",
        "no sé qué más",
        "no entiendo",
        "basta",
        "termina",
    ]

    if any(pattern in response_lower for pattern in evasive_patterns):
        return True

    # Check if response is completely off-topic for the expected field
    if expected_field and len(response.split()) > 2:
        field_config = FIELD_CONFIG.get(expected_field, {})
        field_keywords = field_config.get("keywords", [])
        field_description = field_config.get("description", "").lower()

        # If it's a substantial response but doesn't relate to the field at all
        response_has_field_relation = any(
            keyword.lower() in response_lower for keyword in field_keywords
        ) or any(
            word in response_lower
            for word in field_description.split()
            if len(word) > 3
        )

        if not response_has_field_relation:
            # Check if it's giving information about a different field
            other_field_mentioned = False
            for other_field, other_config in FIELD_CONFIG.items():
                if other_field != expected_field:
                    other_keywords = other_config.get("keywords", [])
                    if any(
                        keyword.lower() in response_lower for keyword in other_keywords
                    ):
                        other_field_mentioned = True
                        break

            # If talking about something completely different, consider evasive
            if not other_field_mentioned:
                return True

    return False


def _compare_anamnesis(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Compara dos estructuras de anamnesis y retorna True si hay nueva información"""

    def _count_filled_fields(anamnesis: Dict[str, Any]) -> int:
        count = 0
        if anamnesis.get("motivo"):
            count += 1

        sympt = anamnesis.get("sintoma_principal", {})
        for field in ["nombre", "inicio", "duracion_horas", "curso", "intensidad_0_10"]:
            if sympt.get(field):
                count += 1

        for field in [
            "antecedentes_personales",
            "antecedentes_familiares",
            "habitos_riesgo",
            "sintomas_asociados",
        ]:
            if anamnesis.get(field):
                count += 1

        return count

    old_count = _count_filled_fields(old)
    new_count = _count_filled_fields(new)
    return new_count > old_count


def sufficiency_gate_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})

    # Track graph steps
    graph_steps = internal.get("graph_steps", 0) + 1
    internal["graph_steps"] = graph_steps

    # EMERGENCY EXIT CONDITIONS - Very aggressive to prevent recursion
    loop_count = internal.get("planner_loop_count", 0)
    conversation_turns = len(data.get("raw_dialog", [])) // 2
    failed_fields_count = len(internal.get("failed_fields", []))

    # Immediate forced sufficiency conditions
    emergency_sufficient = (
        graph_steps > 18  # Too many graph steps
        or loop_count > 4  # Too many planner loops
        or conversation_turns > 10  # Long conversation
        or failed_fields_count > 2  # Too many failed fields
        or internal.get("emergency_exit", False)  # Emergency flag set
    )

    if emergency_sufficient:
        internal["has_sufficiency"] = True
        internal["forced_sufficient"] = True
        internal["emergency_exit_reason"] = {
            "graph_steps": graph_steps,
            "loop_count": loop_count,
            "conversation_turns": conversation_turns,
            "failed_fields_count": failed_fields_count,
        }
        return {"data": data}

    # Enhanced sufficiency evaluation with multiple factors
    anamnesis = data.get("anamnesis", default_anamnesis())
    base_sufficiency = has_sufficient_data(anamnesis)

    # Adaptive thresholds based on conversation context
    force_sufficient = False

    # Factor 1: User seems frustrated or uncooperative
    if _detect_user_frustration(dict(data)):
        force_sufficient = True

    # Factor 2: We have minimal viable info even if not "sufficient"
    if not base_sufficiency and _has_minimal_viable_info(anamnesis):
        # Check if we've made reasonable attempts
        if conversation_turns > 4 or failed_fields_count > 0:
            force_sufficient = True

    # Factor 3: Quality over quantity - if we have good core info
    if _has_high_quality_core_info(anamnesis):
        force_sufficient = True

    # Make final decision
    has_sufficiency = base_sufficiency or force_sufficient
    internal["has_sufficiency"] = has_sufficiency

    # Log the decision factors for debugging
    internal["sufficiency_factors"] = {
        "base_sufficiency": base_sufficiency,
        "loop_count": loop_count,
        "conversation_turns": conversation_turns,
        "failed_fields_count": failed_fields_count,
        "forced_sufficient": force_sufficient,
        "final_decision": has_sufficiency,
    }

    if has_sufficiency:
        internal["planner_loop_count"] = 0

    return {"data": data}


def _detect_user_frustration(data: Dict[str, Any]) -> bool:
    """Detect if user shows signs of frustration"""
    raw_dialog = data.get("raw_dialog", [])
    if len(raw_dialog) < 4:
        return False

    # Look at recent user messages for frustration indicators
    recent_user_msgs = [
        turn.get("content", "").lower()
        for turn in raw_dialog[-6:]
        if turn.get("role") == "user"
    ]

    frustration_patterns = [
        "ya te dije",
        "ya te conté",
        "no sé qué más",
        "basta",
        "termina",
        "no entiendo por qué",
        "ya respondí",
        "no quiero más",
        "esto es muy largo",
        "cuántas preguntas",
    ]

    frustration_count = sum(
        1
        for msg in recent_user_msgs
        for pattern in frustration_patterns
        if pattern in msg
    )

    return frustration_count >= 1


def _has_minimal_viable_info(anamnesis: Dict[str, Any]) -> bool:
    """Check if we have absolute minimum info needed"""
    # Must have complaint
    has_complaint = bool(
        anamnesis.get("motivo") or anamnesis.get("sintoma_principal", {}).get("nombre")
    )

    if not has_complaint:
        return False

    # Must have at least one piece of additional information
    sympt = anamnesis.get("sintoma_principal", {})
    additional_info_count = sum(
        [
            bool(sympt.get("inicio")),
            bool(sympt.get("duracion_horas")),
            bool(sympt.get("intensidad_0_10")),
            bool(sympt.get("curso")),
            bool(anamnesis.get("antecedentes_personales")),
            bool(anamnesis.get("sintomas_asociados")),
        ]
    )

    return additional_info_count >= 1


def _has_high_quality_core_info(anamnesis: Dict[str, Any]) -> bool:
    """Check if we have high-quality core information that's sufficient for basic triage"""
    sympt = anamnesis.get("sintoma_principal", {})

    # High quality means: clear symptom + temporal info + characterization
    has_clear_symptom = bool(anamnesis.get("motivo") and sympt.get("nombre"))
    has_temporal = bool(sympt.get("inicio") or sympt.get("duracion_horas"))
    has_characterization = bool(sympt.get("intensidad_0_10") or sympt.get("curso"))

    return has_clear_symptom and has_temporal and has_characterization


def normalize_struct_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    if data.get("structured") is None:
        structured = call_mcp_tool(
            data,
            "mcp_normalize_struct",
            {"anamnesis": data.get("anamnesis", {})},
            "Normalizar la anamnesis a un formato estándar.",
            lambda payload: mock_normalize(payload.get("anamnesis", {})),
        )
        data["structured"] = structured
    return {"data": data}


def maybe_translate_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})
    needs_translation = CLASSIFIER_TARGET_LANG == "en" and data.get("lang") != "en"
    internal["needs_translation"] = needs_translation
    if needs_translation:
        translated = call_mcp_tool(
            data,
            "mcp_translate",
            {
                "structured": data.get("structured"),
                "from_lang": data.get("lang"),
                "to_lang": CLASSIFIER_TARGET_LANG,
            },
            "Traducción controlada para el clasificador.",
            mock_translate,
        )
        internal["translated_structured"] = translated.get("structured")
    return {"data": data}


def classifier_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    internal = data.setdefault("_internal", {})
    structured_for_model = internal.get("translated_structured") or data.get(
        "structured"
    )
    if structured_for_model is None:
        return {"data": data}
    raw_predictions = call_mcp_tool(
        data,
        "mcp_classifier",
        {
            "structured": structured_for_model,
            "lang": CLASSIFIER_TARGET_LANG
            if internal.get("needs_translation")
            else data.get("lang"),
        },
        "Estimar probabilidades top-3 para orientar recomendaciones.",
        mock_classifier,
    )
    predictions = raw_predictions
    if isinstance(raw_predictions, dict) and raw_predictions.get("probabilities"):
        probabilities = raw_predictions.get("probabilities", [])
        predictions = [
            {
                "condicion": str(item.get("label", "")),
                "prob": float(item.get("prob", 0.0)),
            }
            for item in probabilities
            if isinstance(item, dict)
        ]
        metadata = internal.setdefault("classifier_metadata", {})
        metadata["model_source"] = raw_predictions.get("model_source")
        metadata["features_used"] = raw_predictions.get("features_used", [])
        metadata["prediction"] = raw_predictions.get("prediction")
    data["predicciones"] = predictions
    return {"data": data}


def calibrate_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    if data.get("predicciones"):
        calibrated = call_mcp_tool(
            data,
            "mcp_calibrate",
            {"predicciones": data.get("predicciones")},
            "Reescalar probabilidades para mejor calibración.",
            lambda payload: mock_calibrate(payload.get("predicciones", [])),
        )
        data["predicciones"] = calibrated
    return {"data": data}


def triage_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    if not data.get("structured"):
        return {"data": data}
    triage = call_mcp_tool(
        data,
        "mcp_triage_rules",
        {
            "structured": data.get("structured"),
            "predicciones": data.get("predicciones"),
        },
        "Aplicar reglas de triaje para orientar recomendaciones.",
        mock_triage,
    )
    data["triage"] = triage
    links = call_mcp_tool(
        data,
        "mcp_health_links",
        {"lang": data.get("lang", "es")},
        "Ofrecer recursos confiables en el idioma del usuario.",
        mock_health_links,
    )
    data["resources"] = links
    data["advice"] = triage.get("razon", [])
    return {"data": data}


def present_results_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    anamnesis = data.get("anamnesis", default_anamnesis())
    predictions = data.get("predicciones", [])
    triage = data.get("triage", {})
    resources = data.get("resources", [])
    lines: List[str] = []
    motivo = anamnesis.get("motivo")
    sympt = anamnesis.get("sintoma_principal", {})
    lines.append("Resumen de lo que compartiste:")
    if motivo:
        lines.append(f"- Motivo principal: {motivo}.")
    if sympt.get("nombre"):
        onset_part = f" desde {sympt.get('inicio')}" if sympt.get("inicio") else ""
        lines.append(f"- Síntoma principal: {sympt.get('nombre')}{onset_part}.")
    if sympt.get("duracion_horas"):
        lines.append(f"- Duración aproximada: {sympt.get('duracion_horas')} horas.")
    if sympt.get("curso"):
        lines.append(f"- Curso: {sympt.get('curso')}.")
    lines.append("")
    if predictions:
        lines.append("Posibles causas (probabilidades calibradas):")
        for pred in predictions:
            pct = int(round(pred.get("prob", 0) * 100))
            lines.append(f"- {pred.get('condicion')}: {pct}% aprox.")
        lines.append("")
    triage_level = triage.get("nivel")
    if triage_level:
        lines.append(f"Recomendación principal: {triage_level.upper()}.")
    for rec in data.get("advice", []):
        lines.append(f"- {rec}")
    lines.append("")
    lines.append(
        "Signos de alarma: dolor que aumenta, falta de aire, desmayo, fiebre alta. Si aparecen, busca atención inmediata."
    )
    if resources:
        lines.append("")
        lines.append("Recursos sugeridos:")
        for item in resources:
            title = item.get("title")
            url = item.get("url")
            if title and url:
                lines.append(f"- {title}: {url}")
    lines.append("")
    lines.append("Recuerda: esto no reemplaza la valoración médica profesional.")
    default_message = "\n".join(lines).strip()
    message = default_message
    if llm_enabled():
        public_payload = {k: v for k, v in data.items() if not k.startswith("_")}
        llm_message = generate_final_message(public_payload, data.get("lang", "es"))
        if llm_message:
            llm_message = llm_message.strip()
            if llm_message:
                message = llm_message
    record_chunk_event(data, message)
    raw_dialog = data.get("raw_dialog", [])
    raw_dialog.append({"role": "assistant", "content": message})
    data["raw_dialog"] = raw_dialog
    return {"data": data, "messages": [AIMessage(content=message)]}


def safe_goodbye_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    farewell = (
        "Entendido. Detengo el proceso. Si en algún momento deseas retomar, estaré disponible. "
        "Cuídate."
    )
    record_chunk_event(data, farewell)
    raw_dialog = data.get("raw_dialog", [])
    raw_dialog.append({"role": "assistant", "content": farewell})
    data["raw_dialog"] = raw_dialog
    return {"data": data, "messages": [AIMessage(content=farewell)]}


def urgent_exit_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    urgency = (
        "Detecté datos que pueden ser una urgencia. Acude de inmediato a un servicio de emergencias "
        "o llama al número local de emergencias. Esto no reemplaza la valoración médica profesional."
    )
    record_chunk_event(data, urgency)
    raw_dialog = data.get("raw_dialog", [])
    raw_dialog.append({"role": "assistant", "content": urgency})
    data["raw_dialog"] = raw_dialog
    return {"data": data, "messages": [AIMessage(content=urgency)]}


def await_user_node(state: AgentState) -> AgentState:
    data = clone_data(state.get("data", {}))
    return {"data": data}


__all__ = [
    "ask_user_node",
    "await_user_node",
    "calibrate_node",
    "check_consent_node",
    "classifier_node",
    "detect_language_and_greet_node",
    "maybe_translate_node",
    "normalize_struct_node",
    "parse_and_clean_node",
    "planner_node",
    "present_results_node",
    "red_flags_node",
    "safe_goodbye_node",
    "set_classifier_target_lang",
    "sufficiency_gate_node",
    "triage_node",
    "update_state_node",
    "urgent_exit_node",
]
