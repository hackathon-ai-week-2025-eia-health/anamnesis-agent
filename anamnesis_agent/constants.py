STOP_PHRASES = [
    "no quiero continuar",
    "no quiero seguir",
    "no acepto",
    "stop",
    "cancelar",
    "detente",
    "parar",
    "terminar",
]

FIELD_CONFIG = {
    "motivo": {
        "path": ("anamnesis", "motivo"),
        "description": "el motivo principal de tu consulta",
        "example": "qué te preocupa hoy",
        "keywords": ["motivo principal", "motivo de tu consulta"],
    },
    "sintoma_nombre": {
        "path": ("anamnesis", "sintoma_principal", "nombre"),
        "description": "cuál es el síntoma principal",
        "example": "dolor en el pecho, tos persistente",
        "keywords": ["síntoma principal"],
    },
    "sintoma_inicio": {
        "path": ("anamnesis", "sintoma_principal", "inicio"),
        "description": "desde cuándo comenzó el síntoma",
        "example": "desde ayer, hace dos semanas",
        "keywords": ["desde cuándo", "cuando empezó"],
    },
    "sintoma_duracion_horas": {
        "path": ("anamnesis", "sintoma_principal", "duracion_horas"),
        "description": "cuánto tiempo lleva presente el síntoma",
        "example": "2 horas continuas, 3 días",
        "keywords": ["cuánto tiempo", "duró", "duración"],
    },
    "sintoma_curso": {
        "path": ("anamnesis", "sintoma_principal", "curso"),
        "description": "si el síntoma es constante o intermitente",
        "example": "constante, va y viene",
        "keywords": ["curso", "constante", "intermitente", "va y viene"],
    },
    "sintoma_intensidad": {
        "path": ("anamnesis", "sintoma_principal", "intensidad_0_10"),
        "description": "la intensidad del síntoma en escala 0-10",
        "example": "por ejemplo 7 de 10",
        "keywords": ["escala", "0 al 10", "intensidad"],
    },
    "antecedentes_personales": {
        "path": ("anamnesis", "antecedentes_personales"),
        "description": "tus antecedentes personales relevantes",
        "example": "hipertensión, diabetes",
        "keywords": ["antecedentes personales"],
    },
    "antecedentes_familiares": {
        "path": ("anamnesis", "antecedentes_familiares"),
        "description": "antecedentes de salud en tu familia cercana",
        "example": "infarto en madre, cáncer en hermanos",
        "keywords": ["antecedentes familiares"],
    },
    "habitos_riesgo": {
        "path": ("anamnesis", "habitos_riesgo"),
        "description": "hábitos o exposiciones potencialmente de riesgo",
        "example": "tabaquismo, consumo de alcohol, exposición laboral",
        "keywords": ["hábitos de riesgo", "habitos de riesgo", "consumos", "tabaquismo"],
    },
    "sintomas_asociados": {
        "path": ("anamnesis", "sintomas_asociados"),
        "description": "otros síntomas asociados",
        "example": "fiebre, náuseas, mareos",
        "keywords": ["síntomas asociados", "sintomas asociados"],
    },
    "factores_alivia": {
        "path": ("anamnesis", "sintoma_principal", "factores", "alivia"),
        "description": "qué lo alivia",
        "example": "reposo, analgésicos",
        "keywords": ["lo alivia"],
    },
    "factores_agrava": {
        "path": ("anamnesis", "sintoma_principal", "factores", "agrava"),
        "description": "qué lo agrava",
        "example": "ejercicio, estrés",
        "keywords": ["lo agrava"],
    },
}
