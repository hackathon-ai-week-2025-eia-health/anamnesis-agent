import copy
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from .constants import FIELD_CONFIG


def _append_messages(
    existing: Optional[List[BaseMessage]], new: List[BaseMessage]
) -> List[BaseMessage]:
    base = list(existing or [])
    base.extend(new or [])
    return base


class AgentData(TypedDict, total=False):
    lang: str
    consent: Literal["accepted", "rejected"]
    stop_reason: Optional[Literal["user_declined", "red_flags"]]
    raw_dialog: List[Dict[str, str]]
    anamnesis: Dict[str, Any]
    red_flags: Dict[str, Any]
    structured: Optional[Dict[str, Any]]
    predicciones: List[Dict[str, Any]]
    triage: Dict[str, Any]
    advice: List[str]
    resources: List[Dict[str, str]]
    tool_calls_audit: List[Dict[str, Any]]
    _internal: Dict[str, Any]


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], _append_messages]
    data: AgentData


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_anamnesis() -> Dict[str, Any]:
    return {
        "motivo": None,
        "sintoma_principal": {
            "nombre": None,
            "inicio": None,
            "duracion_horas": None,
            "curso": None,
            "intensidad_0_10": None,
            "factores": {"alivia": [], "agrava": []},
        },
        "antecedentes_personales": [],
        "antecedentes_familiares": [],
        "habitos_riesgo": [],
        "sintomas_asociados": [],
    }


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+", "[correo]", text, flags=re.I)
    cleaned = re.sub(r"\b\d{7,}\b", "[número]", cleaned)
    return cleaned.strip()


def split_items(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r",|;| y | e |\\.\\s", text)
    items: List[str] = []
    for part in parts:
        value = part.strip()
        if value:
            items.append(value)
    return items


def extract_duration_hours(text: str) -> Optional[float]:
    if not text:
        return None

    # Primero intentar con expresiones como "desde ayer", "desde hace 2 días"
    desde_match = re.search(
        r"desde\s+(?:hace\s+)?(\d+(?:[.,]\d+)?)\s*(minuto|minutos|hora|horas|día|días|semana|semanas)",
        text,
        flags=re.I,
    )
    if desde_match:
        value = float(desde_match.group(1).replace(",", "."))
        unit = desde_match.group(2).lower()
        if unit.startswith("minuto"):
            return round(value / 60.0, 2)
        if unit.startswith("hora"):
            return round(value, 2)
        if unit.startswith("día"):
            return round(value * 24.0, 2)
        if unit.startswith("semana"):
            return round(value * 24.0 * 7.0, 2)

    # Si dice "desde ayer" sin número específico, asumir 1 día
    if re.search(r"desde\s+ayer", text, re.I):
        return 24.0

    # Patrón normal para duraciones explícitas
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(minuto|minutos|hora|horas|día|días|semana|semanas)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    if unit.startswith("minuto"):
        return round(value / 60.0, 2)
    if unit.startswith("hora"):
        return round(value, 2)
    if unit.startswith("día"):
        return round(value * 24.0, 2)
    if unit.startswith("semana"):
        return round(value * 24.0 * 7.0, 2)
    return None


def extract_multiple_fields(text: str, anamnesis: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae múltiples campos de una sola respuesta del usuario"""
    extracted = {}
    text_lower = text.lower().strip()

    # Extraer intensidad
    intensity = extract_intensity(text)
    if intensity is not None:
        extracted["sintoma_intensidad"] = intensity

    # Extraer duración
    duration = extract_duration_hours(text)
    if duration is not None:
        extracted["sintoma_duracion_horas"] = duration

    # Extraer curso del síntoma
    course_patterns = {
        r"constante|continuo|continuamente|todo el tiempo|sin parar": "constante",
        r"intermitente|va y viene|a ratos|de vez en cuando|por momentos": "intermitente",
        r"empeorando|cada vez peor|más fuerte": "empeorando",
        r"mejorando|menos|aliviando": "mejorando",
    }

    for pattern, course in course_patterns.items():
        if re.search(pattern, text_lower):
            extracted["sintoma_curso"] = course
            break

    # Extraer factores que alivian o agravan
    if any(word in text_lower for word in ["alivia", "mejora", "calma", "ayuda"]):
        relieving_factors = _extract_factors(text, "alivia")
        if relieving_factors:
            extracted["factores_alivia"] = relieving_factors

    if any(
        word in text_lower for word in ["empeora", "agrava", "aumenta", "duele más"]
    ):
        aggravating_factors = _extract_factors(text, "agrava")
        if aggravating_factors:
            extracted["factores_agrava"] = aggravating_factors

    # Extraer información temporal (inicio)
    if re.search(r"desde|empezó|comenzó|inició", text_lower):
        extracted["sintoma_inicio"] = text.strip()

    return extracted


def _extract_factors(text: str, factor_type: str) -> List[str]:
    """Extrae factores que alivian o agravan el síntoma"""
    factors = []
    text_lower = text.lower()

    # Patrones comunes
    if factor_type == "alivia":
        patterns = [
            r"con\s+(\w+(?:\s+\w+)?)",
            r"cuando\s+(.+?)(?:,|\.|$)",
            r"al\s+(\w+(?:\s+\w+)?)",
            r"tomando\s+(\w+(?:\s+\w+)?)",
        ]
    else:  # agrava
        patterns = [
            r"con\s+el?\s+(\w+(?:\s+\w+)?)",
            r"al\s+(\w+(?:\s+\w+)?)",
            r"cuando\s+(.+?)(?:,|\.|$)",
        ]

    for pattern in patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            factor = match.group(1).strip()
            if factor and factor not in factors:
                factors.append(factor)

    return factors


def extract_intensity(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(\d{1,2})\s*(?:/10|sobre 10|de 0 a 10)", text)
    if match:
        value = int(match.group(1))
        return max(0, min(10, value))
    match = re.search(r"intensidad\s*(\d{1,2})", text)
    if match:
        value = int(match.group(1))
        return max(0, min(10, value))
    return None


def _set_on_path(data: Dict[str, Any], path: tuple, value: Any) -> None:
    current = data
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def _ensure_list_on_path(data: Dict[str, Any], path: tuple) -> List[Any]:
    current = data
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    last = path[-1]
    if last not in current or not isinstance(current[last], list):
        current[last] = []
    return current[last]


def _append_unique(lst: List[str], items: List[str]) -> None:
    for item in items:
        clean = item.strip()
        if clean and clean not in lst:
            lst.append(clean)


def apply_field(anamnesis: Dict[str, Any], field_key: str, text: str) -> None:
    cleaned = clean_text(text)
    config = FIELD_CONFIG.get(field_key)
    if not config or not cleaned:
        return
    path = config["path"]
    if field_key == "sintoma_duracion_horas":
        value = extract_duration_hours(cleaned)
        if value is not None:
            _set_on_path(anamnesis, path, value)
            # Si la respuesta también contiene información de curso, guardarla
            if any(
                word in cleaned.lower()
                for word in ["continuamente", "constante", "intermitente", "va y viene"]
            ):
                sympt = anamnesis.setdefault("sintoma_principal", {})
                if not sympt.get("curso"):
                    if (
                        "continuamente" in cleaned.lower()
                        or "constante" in cleaned.lower()
                    ):
                        sympt["curso"] = "constante"
                    elif (
                        "intermitente" in cleaned.lower()
                        or "va y viene" in cleaned.lower()
                    ):
                        sympt["curso"] = "intermitente"
        return
    if field_key == "sintoma_intensidad":
        value = extract_intensity(cleaned)
        if value is not None:
            _set_on_path(anamnesis, path, value)
        return
    if field_key in {
        "antecedentes_personales",
        "antecedentes_familiares",
        "habitos_riesgo",
        "sintomas_asociados",
    }:
        existing = _ensure_list_on_path(anamnesis, path)
        items = split_items(cleaned)
        if not items:
            items = [cleaned]
        _append_unique(existing, items)
        return
    if field_key in {"factores_alivia", "factores_agrava"}:
        lst = _ensure_list_on_path(anamnesis, path)
        items = split_items(cleaned)
        if not items:
            items = [cleaned]
        _append_unique(lst, items)
        return
    if field_key == "motivo" and not anamnesis.get("motivo"):
        anamnesis["motivo"] = cleaned
        return
    if field_key == "sintoma_nombre":
        sympt = anamnesis.setdefault("sintoma_principal", {})
        if not sympt.get("nombre"):
            sympt["nombre"] = cleaned
        return
    if field_key == "sintoma_inicio":
        sympt = anamnesis.setdefault("sintoma_principal", {})
        sympt["inicio"] = cleaned
        return
    if field_key == "sintoma_curso":
        sympt = anamnesis.setdefault("sintoma_principal", {})
        sympt["curso"] = cleaned
        return
    _set_on_path(anamnesis, path, cleaned)


def user_answered_question(
    user_response: str, expected_field: str, current_anamnesis: Dict[str, Any]
) -> bool:
    """Enhanced determination of whether user response answered the expected question"""
    if not user_response or not expected_field:
        return False

    response_lower = user_response.lower().strip()

    # Enhanced evasive response detection
    evasive_responses = [
        "no sé", "no lo sé", "no estoy seguro", "no recuerdo", "no me acuerdo",
        "no sabría decir", "no tengo idea", "no lo recuerdo", "no sabría",
        "no estoy segura", "no me doy cuenta", "no puedo decir"
    ]

    if any(phrase in response_lower for phrase in evasive_responses):
        return True  # Evasive but still counts as a response

    # Check for field-specific response patterns
    field_responses = _get_field_response_patterns(expected_field)
    
    # Look for positive indicators
    has_field_indicators = any(
        indicator in response_lower for indicator in field_responses.get("indicators", [])
    )
    
    # Look for value patterns
    has_value_patterns = any(
        pattern in response_lower for pattern in field_responses.get("value_patterns", [])
    )
    
    # Check response length and complexity
    word_count = len(response_lower.split())
    
    # Simple responses that might be valid
    if word_count <= 3:
        simple_valid = field_responses.get("simple_valid", [])
        if any(phrase in response_lower for phrase in simple_valid):
            return True
        # Single word/number responses for specific fields
        if expected_field == "sintoma_intensidad" and any(char.isdigit() for char in response_lower):
            return True
        return False
    
    # Complex responses - check for relevance
    if has_field_indicators or has_value_patterns:
        return True
    
    # Check if response seems to contain useful medical information
    if _contains_medical_info(response_lower, expected_field):
        return True
    
    # For longer responses, be more permissive if they seem genuine
    if word_count >= 5:
        # Check if it's a genuine attempt to communicate
        if not _seems_frustrated_or_confused(response_lower):
            return True
    
    return False


def _get_field_response_patterns(field: str) -> Dict[str, List[str]]:
    """Get response patterns for specific fields"""
    patterns = {
        "motivo": {
            "indicators": ["dolor", "molestia", "problema", "síntoma", "siento", "tengo"],
            "value_patterns": ["me duele", "siento", "tengo", "problema con"],
            "simple_valid": ["dolor", "tos", "fiebre", "mareo", "náuseas"]
        },
        "sintoma_nombre": {
            "indicators": ["dolor", "duele", "molestia", "siento", "tengo", "tos", "fiebre"],
            "value_patterns": ["me duele", "dolor en", "dolor de", "tengo", "siento"],
            "simple_valid": ["dolor", "tos", "fiebre", "mareo", "náuseas", "cansancio"]
        },
        "sintoma_inicio": {
            "indicators": ["desde", "empezó", "comenzó", "ayer", "hoy", "hace", "cuando"],
            "value_patterns": ["desde", "hace", "empezó", "comenzó", "ayer", "hoy"],
            "simple_valid": ["ayer", "hoy", "anoche", "mañana"]
        },
        "sintoma_duracion_horas": {
            "indicators": ["horas", "días", "minutos", "tiempo", "desde", "hace", "llevo"],
            "value_patterns": ["horas", "días", "minutos", "hace", "desde", "llevo", "tiempo"],
            "simple_valid": ["ayer", "hoy", "horas", "días"]
        },
        "sintoma_curso": {
            "indicators": ["constante", "continuo", "intermitente", "va y viene", "siempre", "a ratos"],
            "value_patterns": ["constante", "continuo", "intermitente", "va y viene", "todo el tiempo"],
            "simple_valid": ["constante", "continuo", "intermitente", "siempre"]
        },
        "sintoma_intensidad": {
            "indicators": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "fuerte", "leve"],
            "value_patterns": ["/10", "de 10", "del 0", "intensidad", "fuerte", "leve", "moderado"],
            "simple_valid": ["fuerte", "leve", "mucho", "poco", "moderado"]
        },
        "antecedentes_personales": {
            "indicators": ["diabetes", "hipertensión", "medicamento", "enfermedad", "problema", "tomo"],
            "value_patterns": ["tengo", "tomo", "medicamento", "pastillas", "diabetes", "presión"],
            "simple_valid": ["nada", "ninguno", "no", "diabetes", "hipertensión"]
        },
        "antecedentes_familiares": {
            "indicators": ["familia", "padre", "madre", "hermano", "familiar", "papá", "mamá"],
            "value_patterns": ["familia", "padre", "madre", "hermano", "familiar", "abuelo"],
            "simple_valid": ["nada", "nadie", "no", "ninguno"]
        },
        "habitos_riesgo": {
            "indicators": ["fumo", "fumar", "alcohol", "beber", "trabajo", "exposición"],
            "value_patterns": ["fumo", "bebo", "trabajo", "alcohol", "cigarrillos", "exposición"],
            "simple_valid": ["nada", "no", "ninguno", "fumo", "bebo"]
        },
        "sintomas_asociados": {
            "indicators": ["también", "además", "acompañado", "junto", "otro", "más"],
            "value_patterns": ["también", "además", "acompañado", "junto con", "otro síntoma"],
            "simple_valid": ["nada", "no", "ninguno", "solo", "únicamente"]
        }
    }
    
    return patterns.get(field, {
        "indicators": [],
        "value_patterns": [],
        "simple_valid": []
    })


def _contains_medical_info(response: str, expected_field: str) -> bool:
    """Check if response contains medical information relevant to any field"""
    medical_terms = [
        "dolor", "molestia", "síntoma", "enfermedad", "medicamento", "pastilla",
        "diabetes", "hipertensión", "presión", "corazón", "cabeza", "estómago",
        "pecho", "espalda", "pierna", "brazo", "tos", "fiebre", "mareo",
        "náuseas", "vómito", "diarrea", "estreñimiento", "cansancio"
    ]
    
    return any(term in response for term in medical_terms)


def _seems_frustrated_or_confused(response: str) -> bool:
    """Detect if response shows frustration or confusion"""
    frustration_patterns = [
        "ya te dije", "ya te conté", "no entiendo", "qué quieres",
        "no sé qué", "qué más", "esto es", "por qué"
    ]
    
    return any(pattern in response for pattern in frustration_patterns)


def guess_field_from_assistant(text: str) -> Optional[str]:
    lowered = text.lower()
    for key, config in FIELD_CONFIG.items():
        for keyword in config.get("keywords", []):
            if keyword in lowered:
                return key
    return None
    lowered = text.lower()
    for key, config in FIELD_CONFIG.items():
        for keyword in config.get("keywords", []):
            if keyword in lowered:
                return key
    return None


def parse_dialog(raw_dialog: List[Dict[str, str]]) -> Dict[str, Any]:
    anamnesis = default_anamnesis()
    pending_field: Optional[str] = None
    latest_user = ""

    for turn in raw_dialog:
        role = turn.get("role")
        content = clean_text(turn.get("content", ""))

        if role == "assistant":
            maybe_field = guess_field_from_assistant(content)
            if maybe_field:
                pending_field = maybe_field
        elif role == "user":
            latest_user = content

            # Extraer múltiples campos de la respuesta
            extracted_fields = extract_multiple_fields(content, anamnesis)

            if pending_field:
                # Verificar si realmente respondió la pregunta
                did_answer = user_answered_question(content, pending_field, anamnesis)

                if did_answer:
                    # Aplicar el campo específico que se estaba preguntando
                    apply_field(anamnesis, pending_field, content)
                    pending_field = None  # Limpiar pending field solo si respondió
                # Si no respondió, mantener pending_field para continuar preguntando

            # Aplicar todos los campos adicionales extraídos
            for field_key, value in extracted_fields.items():
                if field_key == "sintoma_intensidad" and not anamnesis[
                    "sintoma_principal"
                ].get("intensidad_0_10"):
                    anamnesis["sintoma_principal"]["intensidad_0_10"] = value
                elif field_key == "sintoma_duracion_horas" and not anamnesis[
                    "sintoma_principal"
                ].get("duracion_horas"):
                    anamnesis["sintoma_principal"]["duracion_horas"] = value
                elif field_key == "sintoma_curso" and not anamnesis[
                    "sintoma_principal"
                ].get("curso"):
                    anamnesis["sintoma_principal"]["curso"] = value
                elif field_key == "sintoma_inicio" and not anamnesis[
                    "sintoma_principal"
                ].get("inicio"):
                    anamnesis["sintoma_principal"]["inicio"] = value
                elif field_key == "factores_alivia":
                    existing = (
                        anamnesis["sintoma_principal"]
                        .setdefault("factores", {})
                        .setdefault("alivia", [])
                    )
                    _append_unique(
                        existing, value if isinstance(value, list) else [value]
                    )
                elif field_key == "factores_agrava":
                    existing = (
                        anamnesis["sintoma_principal"]
                        .setdefault("factores", {})
                        .setdefault("agrava", [])
                    )
                    _append_unique(
                        existing, value if isinstance(value, list) else [value]
                    )

            # Lógica de fallback para campos básicos si no hay pending_field
            if not pending_field:
                if not anamnesis.get("motivo") and len(content) > 10:
                    # Si es una descripción larga, probablemente sea el motivo
                    anamnesis["motivo"] = content
                elif not anamnesis["sintoma_principal"].get("nombre"):
                    # Buscar patrones de síntomas
                    symptom_patterns = [
                        r"me duele (la|el) (\w+(?:\s+\w+)?)",
                        r"tengo (dolor|tos|fiebre|náuseas|mareo)(?:\s+(?:en|de)\s+(\w+))?",
                        r"siento (\w+(?:\s+\w+)?)",
                    ]
                    for pattern in symptom_patterns:
                        match = re.search(pattern, content.lower())
                        if match:
                            if len(match.groups()) > 1 and match.group(2):
                                anamnesis["sintoma_principal"]["nombre"] = (
                                    f"{match.group(1)} {match.group(2)}"
                                )
                            else:
                                anamnesis["sintoma_principal"]["nombre"] = match.group(
                                    1
                                )
                            break
    return {
        "anamnesis": anamnesis,
        "pending_field": pending_field,
        "latest_user_text": latest_user,
    }


def ensure_parsed(data: AgentData) -> Dict[str, Any]:
    internal = data.setdefault("_internal", {})
    raw_dialog = data.get("raw_dialog", [])
    digest = tuple((turn.get("role"), turn.get("content")) for turn in raw_dialog)
    if (
        internal.get("parsed_dialog") is None
        or internal.get("parsed_raw_len") != len(raw_dialog)
        or internal.get("parsed_raw_digest") != digest
    ):
        parsed = parse_dialog(raw_dialog)
        internal["parsed_dialog"] = parsed
        internal["latest_user_text"] = parsed.get("latest_user_text", "")
        internal["awaiting_field"] = parsed.get("pending_field")
        internal["parsed_raw_len"] = len(raw_dialog)
        internal["parsed_raw_digest"] = digest
    return internal["parsed_dialog"]


def has_sufficient_data(anamnesis: Dict[str, Any]) -> bool:
    """Enhanced determination with more permissive thresholds to prevent loops"""
    # Must have basic complaint
    if not anamnesis.get("motivo") and not anamnesis["sintoma_principal"].get("nombre"):
        return False

    sympt = anamnesis["sintoma_principal"]
    
    # More permissive approach - if we have complaint + ANY additional info, that's often enough
    additional_info = [
        sympt.get("inicio"),
        sympt.get("duracion_horas"), 
        sympt.get("curso"),
        sympt.get("intensidad_0_10"),
        anamnesis.get("antecedentes_personales"),
        anamnesis.get("sintomas_asociados")
    ]
    
    # Count non-empty additional info
    info_count = sum(1 for info in additional_info if info)
    
    # Very permissive threshold - just 2 pieces of additional info
    if info_count >= 2:
        return True
    
    # Even more permissive for basic scenarios
    # If we have temporal info (inicio OR duracion) + any characterization, that's enough
    has_temporal = bool(sympt.get("inicio") or sympt.get("duracion_horas"))
    has_characterization = bool(sympt.get("intensidad_0_10") or sympt.get("curso"))
    
    if has_temporal and has_characterization:
        return True
    
    # If we have temporal + medical context, that's sufficient too
    has_medical_context = bool(anamnesis.get("antecedentes_personales") or anamnesis.get("sintomas_asociados"))
    if has_temporal and has_medical_context:
        return True
    
    # Last resort - if we have intensity + any other info
    if sympt.get("intensidad_0_10") and info_count >= 1:
        return True
        
    return False


def determine_missing_fields(parsed: Dict[str, Any]) -> List[str]:
    """Enhanced determination of missing fields with intelligent prioritization"""
    anamnesis = parsed.get("anamnesis", default_anamnesis())
    missing: List[str] = []

    # Absolute critical fields (always needed)
    if not anamnesis.get("motivo"):
        missing.append("motivo")
    if not anamnesis["sintoma_principal"].get("nombre"):
        missing.append("sintoma_nombre")

    # Stop here if we don't have basics - don't overwhelm user
    if missing:
        return missing

    # Temporal information - flexible approach
    sympt = anamnesis["sintoma_principal"]
    has_temporal_info = bool(sympt.get("inicio") or sympt.get("duracion_horas"))

    if not has_temporal_info:
        # Prioritize duration as it's often easier to answer
        missing.append("sintoma_duracion_horas")
    else:
        # If we have one, maybe get the other but not critical
        if not sympt.get("duracion_horas") and len(missing) == 0:
            missing.append("sintoma_duracion_horas")
        elif not sympt.get("inicio") and len(missing) == 0:
            missing.append("sintoma_inicio")

    # Don't ask for more than 2-3 things at once
    if len(missing) >= 2:
        return missing

    # Symptom characterization - smart selection
    if not sympt.get("intensidad_0_10") and not sympt.get("curso"):
        # Intensity is usually easier than course
        missing.append("sintoma_intensidad")
    elif not sympt.get("intensidad_0_10") and len(missing) < 2:
        missing.append("sintoma_intensidad")
    elif not sympt.get("curso") and len(missing) < 2:
        missing.append("sintoma_curso")

    if len(missing) >= 2:
        return missing

    # Medical context - adaptive approach
    has_any_history = bool(
        anamnesis.get("antecedentes_personales") or 
        anamnesis.get("antecedentes_familiares")
    )
    
    if not has_any_history:
        # Start with personal history as it's more relevant
        missing.append("antecedentes_personales")
    
    if len(missing) >= 2:
        return missing

    # Additional symptoms - only if we have good basic info
    if not anamnesis.get("sintomas_asociados") and len(missing) < 2:
        # Only ask if we have solid foundation
        temporal_score = sum([
            bool(sympt.get("inicio")),
            bool(sympt.get("duracion_horas"))
        ])
        characterization_score = sum([
            bool(sympt.get("intensidad_0_10")),
            bool(sympt.get("curso"))
        ])
        
        if temporal_score >= 1 and characterization_score >= 1:
            missing.append("sintomas_asociados")

    if len(missing) >= 2:
        return missing

    # Risk factors - only ask if conversation is going well
    if not anamnesis.get("habitos_riesgo") and len(missing) < 2:
        # Only if we have substantial information already
        info_completeness = _calculate_info_completeness(anamnesis)
        if info_completeness > 0.6:  # 60% complete
            missing.append("habitos_riesgo")

    # Factors (what makes it better/worse) - advanced level
    factors = sympt.get("factores", {})
    if not factors.get("alivia") and not factors.get("agrava") and len(missing) < 2:
        # Only if we're nearly complete
        info_completeness = _calculate_info_completeness(anamnesis)
        if info_completeness > 0.8:  # 80% complete
            missing.append("factores_alivia")

    return missing


def _calculate_info_completeness(anamnesis: Dict[str, Any]) -> float:
    """Calculate how complete the anamnesis information is (0.0 to 1.0)"""
    total_possible = 10
    current_score = 0
    
    # Basic info (4 points possible)
    if anamnesis.get("motivo"):
        current_score += 1
    if anamnesis["sintoma_principal"].get("nombre"):
        current_score += 1
    if anamnesis["sintoma_principal"].get("inicio") or anamnesis["sintoma_principal"].get("duracion_horas"):
        current_score += 1
    if anamnesis["sintoma_principal"].get("intensidad_0_10") or anamnesis["sintoma_principal"].get("curso"):
        current_score += 1
    
    # Additional info (6 points possible)
    if anamnesis.get("antecedentes_personales"):
        current_score += 1
    if anamnesis.get("antecedentes_familiares"):
        current_score += 1
    if anamnesis.get("sintomas_asociados"):
        current_score += 1
    if anamnesis.get("habitos_riesgo"):
        current_score += 1
    
    # Detailed symptom info (2 points)
    if anamnesis["sintoma_principal"].get("intensidad_0_10") and anamnesis["sintoma_principal"].get("curso"):
        current_score += 1
    if anamnesis["sintoma_principal"].get("inicio") and anamnesis["sintoma_principal"].get("duracion_horas"):
        current_score += 1
    
    return min(current_score / total_possible, 1.0)


def detect_language_from_dialog(
    raw_dialog: List[Dict[str, str]], fallback: str = "es"
) -> str:
    text = " ".join(turn.get("content", "") for turn in raw_dialog).lower()
    if not text:
        return fallback
    spanish_score = sum(
        1
        for token in [" el ", " la ", " de ", " que ", " es ", " dolor "]
        if token in f" {text} "
    )
    english_score = sum(
        1 for token in [" the ", " and ", " pain ", " chest "] if token in f" {text} "
    )
    return "en" if english_score > spanish_score else "es"


def build_initial_data(messages: List[BaseMessage]) -> AgentData:
    raw_dialog: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            raw_dialog.append({"role": "user", "content": msg.content})
        else:
            raw_dialog.append({"role": "assistant", "content": msg.content})
    data: AgentData = {
        "lang": "es",
        "consent": "accepted",
        "stop_reason": None,
        "raw_dialog": raw_dialog,
        "anamnesis": default_anamnesis(),
        "red_flags": {"present": False, "triggers": []},
        "structured": None,
        "predicciones": [],
        "triage": {"nivel": None, "razon": []},
        "advice": [],
        "resources": [],
        "tool_calls_audit": [],
        "_internal": {
            "latest_user_text": "",
            "awaiting_field": None,
            "planner_action": None,
            "planner_target": None,
            "ws_events": [],
            "event_seq": 0,
            "needs_translation": False,
            "translated_structured": None,
            "parsed_dialog": None,
            "has_sufficiency": False,
            "uuid": None,
        },
    }
    return data


def next_event_seq(internal: Dict[str, Any]) -> int:
    seq = internal.get("event_seq", 0)
    internal["event_seq"] = seq + 1
    return seq


def record_chunk_event(data: AgentData, text: str) -> None:
    internal = data.setdefault("_internal", {})
    seq = next_event_seq(internal)
    internal.setdefault("ws_events", []).append(
        {"type": "chunk", "data": text, "seq": seq}
    )


def safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(payload, default=str))
    except Exception:
        return payload


def clone_data(data: AgentData) -> AgentData:
    return copy.deepcopy(data)


__all__ = [
    "AgentData",
    "AgentState",
    "build_initial_data",
    "clean_text",
    "clone_data",
    "now_iso",
    "default_anamnesis",
    "detect_language_from_dialog",
    "determine_missing_fields",
    "ensure_parsed",
    "has_sufficient_data",
    "next_event_seq",
    "record_chunk_event",
    "safe_payload",
]
