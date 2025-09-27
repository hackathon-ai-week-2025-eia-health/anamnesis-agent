"""Bedrock-backed helpers for optional LLM generations."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage


def _env_flag(name: str, default: str = "") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "on", "yes"}


def llm_enabled() -> bool:
    return _env_flag("ANAMNESIS_USE_BEDROCK_LLM")


@lru_cache(maxsize=1)
def get_chat() -> ChatBedrock:
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return ChatBedrock(model=model_id, temperature=temperature, streaming=False)


def _call_bedrock(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        chat = get_chat()
        result = chat.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        if isinstance(result.content, list):
            return "".join(
                part["text"]
                for part in result.content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(result.content)
    except Exception:
        return None


def generate_question(
    field_key: str,
    field_config: Dict[str, Any],
    anamnesis: Dict[str, Any],
    lang: str,
    attempt: int = 1,
) -> Optional[str]:
    if not llm_enabled():
        return None

    attempt = max(1, attempt)

    # Sistema más inteligente que adapta las preguntas al contexto
    system_prompt = (
        "Eres un médico virtual experto. Haces preguntas naturales, empáticas y contextuales para completar una anamnesis. "
        "Adapta tu estilo según la información ya conocida. Sé conversacional pero profesional. "
        "Evita repetir información ya mencionada. Haz una sola pregunta específica y directa."
    )

    # Preparar contexto más rico
    symptom = anamnesis.get("sintoma_principal", {})
    symptom_name = symptom.get("nombre", "el síntoma")

    context_info = []
    if anamnesis.get("motivo"):
        context_info.append(f"Motivo: {anamnesis['motivo']}")
    if symptom.get("nombre"):
        context_info.append(f"Síntoma principal: {symptom['nombre']}")
    if symptom.get("inicio"):
        context_info.append(f"Inicio: {symptom['inicio']}")
    if symptom.get("duracion_horas"):
        context_info.append(f"Duración: {symptom['duracion_horas']} horas")

    context_str = " | ".join(context_info) if context_info else "Primera consulta"

    # Prompts específicos por tipo de campo
    field_prompts = {
        "motivo": "El paciente acaba de empezar la consulta. Pregúntale de manera empática cuál es el motivo principal de su visita.",
        "sintoma_nombre": f"Ya conoces el motivo: '{anamnesis.get('motivo', '')}'. Ahora necesitas que el paciente describa específicamente cuál es su síntoma principal.",
        "sintoma_inicio": f"El paciente tiene {symptom_name}. Pregúntale cuándo comenzó este síntoma de manera natural.",
        "sintoma_duracion_horas": f"El paciente tiene {symptom_name} desde {symptom.get('inicio', 'hace tiempo')}. Pregúntale por cuánto tiempo ha estado presente o cuánto tiempo lleva con él.",
        "sintoma_curso": f"El paciente tiene {symptom_name}. Pregúntale si el síntoma es constante o si va y viene (intermitente).",
        "sintoma_intensidad": f"El paciente tiene {symptom_name}. Pregúntale qué tan intenso es el síntoma en una escala del 0 al 10.",
        "antecedentes_personales": "Necesitas conocer los antecedentes médicos del paciente. Pregúntale si tiene algún problema de salud previo o toma medicamentos.",
        "antecedentes_familiares": "Pregúntale al paciente sobre problemas de salud importantes en su familia cercana (padres, hermanos).",
        "habitos_riesgo": f"Pregúntale sobre hábitos que podrían estar relacionados con su {symptom_name} (fumar, beber, exposiciones laborales, etc.).",
        "sintomas_asociados": f"El paciente tiene {symptom_name}. Pregúntale si ha notado otros síntomas que acompañen o se relacionen con este problema.",
    }

    specific_prompt = field_prompts.get(field_key, field_config.get("description", ""))

    user_prompt = (
        f"Idioma: {lang}\n"
        f"Contexto conocido: {context_str}\n"
        f"Tarea: {specific_prompt}\n\n"
        f"Genera una pregunta natural y empática. No uses formulismos como 'Para avanzar necesito saber...' "
        f"Sé directo y conversacional como un médico real."
    )

    return _call_bedrock(system_prompt, user_prompt)


def generate_final_message(data: Dict[str, Any], lang: str) -> Optional[str]:
    if not llm_enabled():
        return None
    system_prompt = (
        "Eres un asistente clínico digital. Resume de forma ética, neutral y clara. "
        "Incluye recordatorio de que no reemplaza consulta profesional."
    )
    user_prompt = (
        f"Idioma: {lang}.\n"
        "Genera el texto final del resultado a partir del siguiente estado JSON: \n"
        f"{json.dumps(data, ensure_ascii=False)}"
    )
    return _call_bedrock(system_prompt, user_prompt)


__all__ = ["llm_enabled", "generate_question", "generate_final_message"]
