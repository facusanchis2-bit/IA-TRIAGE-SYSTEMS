"""
Triage Inteligente – Microservicio (FastAPI)
=================================================

Requisitos (pip):
    pip install fastapi uvicorn pydantic[dotenv] python-multipart
    pip install spacy scispacy
    # (opcional) modelos spaCy: python -m spacy download es_core_news_sm
    pip install xgboost lightgbm shap numpy pandas scikit-learn
    pip install sqlitedict python-dotenv

Ejecución local:
    export UVICORN_PORT=8000
    uvicorn triage_ai_microservice:app --reload --port ${UVICORN_PORT:-8000}

Estructura de artefactos (opcional):
    ./artifacts/model.pkl   -> modelo entrenado (LightGBM/XGBoost scikit interface)
    ./artifacts/feature_list.json

Notas de privacidad:
    - Se pseudonimizan campos de paciente por defecto. Para almacenar PHI sin hash, definir STORE_RAW_PHI=true
    - Minimización: se guarda el estado esencial; textos libres se acotan y se pueden anonimizar.

"""
from __future__ import annotations
import os
import re
import json
import uuid
import time
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sqlite3, json
from io import BytesIO
import pandas as pd
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML opcional (fall-back si no están instalados)
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

# spaCy / scispaCy
try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

# ---------- Configuración y utilidades ----------

MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
PIPELINE_FINGERPRINT = hashlib.sha256(
    ("nlp:spacy+regex|model:gbm_fallback|rules:v1|explain:shap_optional").encode()
).hexdigest()[:12]

DISCLAIMER_TEXT = (
    "Esta recomendación es asistida por IA y NO reemplaza la evaluación profesional. "
    "Ante síntomas graves, busque atención médica inmediata."
)

DB_PATH = os.getenv("TRIAGE_DB_PATH", "triage_sessions.db")
STORE_RAW_PHI = os.getenv("STORE_RAW_PHI", "false").lower() in {"1", "true", "yes"}
ALLOW_OPENAI = os.getenv("ALLOW_OPENAI", "false").lower() in {"1", "true", "yes"}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CORS básico (ajusta en prod)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ---------- Helpers ----------

ISO = "%Y-%m-%dT%H:%M:%SZ"

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def new_id(prefix: str = "sess") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def pseudonymize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if STORE_RAW_PHI:
        return value
    # hash con sal derivada del fingerprint del pipeline
    return sha256(f"{value}|{PIPELINE_FINGERPRINT}")


def redact_med_doses(text: str) -> str:
    """Elimina dosis explícitas tipo 'mg', 'mcg', 'ml', etc."""
    if not text:
        return text
    pattern = r"\b(\d+(?:[.,]\d+)?)\s*(mg|mcg|ug|g|ml|amp|ampolla|gotas|ui|iu)\b"
    return re.sub(pattern, "[DOSIS REDACTADA]", text, flags=re.IGNORECASE)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ---------- Sesiones y Auditoría (SQLite) ----------

class SessionStore:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    ts TEXT,
                    model_version TEXT,
                    pipeline_hash TEXT,
                    input_hash TEXT,
                    output_hash TEXT,
                    disclaimer_ack INTEGER,
                    endpoint TEXT
                )
                """
            )
            conn.commit()

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        payload = json.dumps(state, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "REPLACE INTO sessions (session_id, data_json, updated_at) VALUES (?, ?, ?)",
                (session_id, payload, now_utc()),
            )
            conn.commit()

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT data_json FROM sessions WHERE session_id=?", (session_id,))
            row = c.fetchone()
            if not row:
                return None
            return json.loads(row[0])

    def log_audit(
        self,
        request_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        input_obj: Dict[str, Any],
        output_obj: Dict[str, Any],
        endpoint: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO audit_log (
                    request_id, user_id, session_id, ts, model_version, pipeline_hash,
                    input_hash, output_hash, disclaimer_ack, endpoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    user_id,
                    session_id,
                    now_utc(),
                    MODEL_VERSION,
                    PIPELINE_FINGERPRINT,
                    sha256(json.dumps(input_obj, ensure_ascii=False)),
                    sha256(json.dumps(output_obj, ensure_ascii=False)),
                    int(bool(output_obj.get("generative_summary", {}).get("disclaimer"))),
                    endpoint,
                ),
            )
            conn.commit()


store = SessionStore()


# ---------- NLP Clínico (ligero + fallbacks) ----------

SPANISH_SYMPTOMS = [
    "dolor", "pecho", "torác", "abdomen", "estómago", "cabeza", "cefalea",
    "disnea", "sudoración", "náuseas", "vomito", "vómito", "fiebre", "hemorragia",
    "desmayo", "mareo", "palpitaciones", "debilidad", "hormigueo"
]

LOCATION_TERMS = [
    "retroesternal", "izquierdo", "derecho", "epigastrio", "hipocondrio", "fosa iliaca",
    "occipital", "temporal", "frontal", "mandíbula", "brazo", "espalda", "hombro"
]

DRUG_TERMS = ["aspirina", "ibuprofeno", "paracetamol", "metformina", "enalapril", "losartan", "atorvastatina"]
CONDITION_TERMS = ["hta", "hipertensión", "dm2", "diabetes", "asma", "epoc", "cardiopatía", "crohn"]


class ClinicalNLP:
    def __init__(self) -> None:
        self.nlp = None
        # Intenta cargar spaCy español
        if spacy is not None:
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except Exception:
                self.nlp = spacy.blank("es")

    def normalize(self, text: str) -> str:
        text = text or ""
        text = text.strip().lower()
        # Quitar acentos simples
        repl = str.maketrans("áéíóúüñ", "aeiouun")
        return text.translate(repl)

    def extract(self, text: str) -> Dict[str, Any]:
        ntext = self.normalize(text)

        # Duración -> horas
        dur_hours = None
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(min|minuto|minutos|h|hora|horas|d|dia|dias)", ntext):
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2)
            if unit.startswith("min"):
                dur_hours = (dur_hours or 0) + val / 60.0
            elif unit.startswith("h") or unit.startswith("hora"):
                dur_hours = (dur_hours or 0) + val
            else:  # día(s)
                dur_hours = (dur_hours or 0) + val * 24

        symptoms = [w for w in SPANISH_SYMPTOMS if w in ntext]
        locations = [w for w in LOCATION_TERMS if w in ntext]
        drugs = [w for w in DRUG_TERMS if w in ntext]
        conditions = [w for w in CONDITION_TERMS if w in ntext]

        radiation_left_arm = any(k in ntext for k in ["brazo izquierdo", "hombro izquierdo", "mandibula izquierda"])
        chest_pain = any(k in ntext for k in ["dolor de pecho", "dolor en el pecho", "dolor torac", "pecho", "torac"]) and "dolor" in ntext
        dyspnea = "disnea" in ntext or "falta de aire" in ntext
        sweating = "sudor" in ntext
        nausea = "nause" in ntext or "vomit" in ntext
        massive_bleeding = "hemorragia" in ntext and ("abundante" in ntext or "masiva" in ntext)
        neuro_deficit = any(k in ntext for k in ["debilidad", "paralisis", "parálisis", "disartria", "afasia", "asimetria facial", "asimetría facial"]) and ("inicio subit" in ntext or "de repente" in ntext)

        return {
            "duration_hours": dur_hours,
            "symptoms": symptoms,
            "locations": locations,
            "drugs": drugs,
            "conditions": conditions,
            "flags": {
                "radiation_left_arm": radiation_left_arm,
                "chest_pain": chest_pain,
                "dyspnea": dyspnea,
                "sweating": sweating,
                "nausea": nausea,
                "massive_bleeding": massive_bleeding,
                "neuro_deficit": neuro_deficit,
            },
        }


# ---------- Árbol de decisión adaptativo ----------

class DecisionTreeManager:
    ROOT = "root"
    IDENT = "identificacion"
    CC = "chief_complaint"
    ADAPT = "adaptive_followups"
    MEDS = "medicacion"
    PMH = "antecedentes"
    RISK = "factores_riesgo"
    CONSENT = "consentimiento"
    DONE = "final"

    def first_question(self) -> Tuple[str, str]:
        return (self.IDENT, "Para empezar, ¿podrías indicar Nombre y Apellido, edad, ciudad y ocupación?")

    def next(self, state: Dict[str, Any]) -> Tuple[str, str]:
        answers = state.get("answers", {})
        if self.IDENT not in answers:
            return (self.IDENT, "Para empezar, ¿podrías indicar Nombre y Apellido, edad, ciudad y ocupación?")
        if self.CC not in answers:
            return (self.CC, "¿Cuál es el motivo principal de consulta? (texto libre)")
        if self.ADAPT not in answers:
            cc_text = answers.get(self.CC, {}).get("text", "")
            return (self.ADAPT, self._adapt_question(cc_text))
        if self.MEDS not in answers:
            return (self.MEDS, "¿Qué medicación tomás actualmente? (genéricos si es posible)")
        if self.PMH not in answers:
            return (self.PMH, "¿Antecedentes médicos relevantes? (HTA, DM2, asma, cardiopatía, etc.)")
        if self.RISK not in answers:
            return (self.RISK, "Factores de riesgo: tabaquismo, antecedentes familiares, etc.")
        if not state.get("disclaimer_ack"):
            return (self.CONSENT, "Confirmá que comprendés el descargo: '" + DISCLAIMER_TEXT + "' ¿Aceptás? (sí/no)")
        return (self.DONE, "Fin del cuestionario.")

    def _adapt_question(self, cc_text: str) -> str:
        nlp = ClinicalNLP()
        ents = nlp.extract(cc_text)
        flags = ents["flags"]
        cc = cc_text.lower()
        if any(k in cc for k in ["pecho", "torac"]):
            return (
                "Sobre el dolor torácico: ¿inicio (súbito/progresivo) y duración?, ¿carácter (opresivo/punzante)?, "
                "¿irradiación (brazo izquierdo/mandíbula)?, ¿disnea?, ¿sudoración?, ¿náuseas?"
            )
        if any(k in cc for k in ["abdomen", "estomago", "estómago"]):
            return (
                "Sobre dolor abdominal: ¿localización (epigastrio/FID/FII/etc.)?, ¿fiebre?, ¿vómitos?, ¿diarrea?, ¿ritmo intestinal?, ¿orina?, ¿última menstruación si aplica?"
            )
        if any(k in cc for k in ["cabeza", "cefalea"]):
            return (
                "Sobre la cefalea: ¿inicio súbito?, ¿rigidez de cuello?, ¿focalidad neurológica?, ¿fotofobia?, ¿fiebre?, ¿peor cefalea de la vida?"
            )
        # fallback: follow-ups genéricos
        _ = flags  # no usado extra
        return "Contame más: inicio, intensidad (0-10), localización, factores desencadenantes, síntomas asociados."


# ---------- Feature Engineering ----------

class FeatureEngineer:
    FEATURE_ORDER = [
        "age", "chest_pain", "dyspnea", "radiation_left_arm", "sweating", "nausea",
        "duration_hours", "massive_bleeding", "neuro_deficit",
        # placeholders para signos vitales (si se integran luego)
        "hr", "sbp", "dbp", "spo2"
    ]

    def build(self, state: Dict[str, Any]) -> Dict[str, float]:
        feats: Dict[str, float] = {k: 0.0 for k in self.FEATURE_ORDER}
        # edad
        age = safe_int(state.get("patient", {}).get("age"))
        feats["age"] = float(age)
        ents = state.get("extracted_entities", {})
        flags = ents.get("flags", {})
        feats.update({
            "chest_pain": 1.0 if flags.get("chest_pain") else 0.0,
            "dyspnea": 1.0 if flags.get("dyspnea") else 0.0,
            "radiation_left_arm": 1.0 if flags.get("radiation_left_arm") else 0.0,
            "sweating": 1.0 if flags.get("sweating") else 0.0,
            "nausea": 1.0 if flags.get("nausea") else 0.0,
            "massive_bleeding": 1.0 if flags.get("massive_bleeding") else 0.0,
            "neuro_deficit": 1.0 if flags.get("neuro_deficit") else 0.0,
            "duration_hours": float(ents.get("duration_hours") or 0.0),
        })
        # vitales (si no hay, quedan en 0)
        vitals = state.get("answers", {}).get("vital_signs", {})
        for k in ["hr", "sbp", "dbp", "spo2"]:
            v = vitals.get(k)
            if v is not None:
                feats[k] = float(v)
        return feats


# ---------- Clasificador (LightGBM/XGBoost con fallback) ----------

ESI_CLASSES = [1, 2, 3, 4, 5]
ESI_TO_MANCHESTER = {
    1: "Red (Inmediato)",
    2: "Orange (Muy urgente)",
    3: "Yellow (Urgente)",
    4: "Green (Menos urgente)",
    5: "Blue (No urgente)",
}


class ESIClassifier:
    def __init__(self) -> None:
        self.model = None
        self.feature_list: Optional[List[str]] = None
        self._load()

    def _load(self) -> None:
        model_path = os.path.join("artifacts", "model.pkl")
        feature_list_path = os.path.join("artifacts", "feature_list.json")
        if joblib and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                if os.path.exists(feature_list_path):
                    with open(feature_list_path, "r", encoding="utf-8") as f:
                        self.feature_list = json.load(f)
            except Exception:
                self.model = None

    def predict_proba(self, features: Dict[str, float]) -> np.ndarray:
        if self.model is not None:
            X = self._features_to_array(features)
            try:
                proba = self.model.predict_proba(X)
                return np.asarray(proba).reshape(1, -1)
            except Exception:
                pass
        # Fallback heurístico (no entrenado)
        score = 0.0
        score += 1.5 * features.get("chest_pain", 0)
        score += 1.4 * features.get("dyspnea", 0)
        score += 1.2 * features.get("radiation_left_arm", 0)
        score += 0.8 * features.get("sweating", 0)
        score += 0.7 * features.get("nausea", 0)
        score += 1.6 * features.get("massive_bleeding", 0)
        score += 1.6 * features.get("neuro_deficit", 0)
        score += 0.05 * max(0.0, 10.0 - features.get("duration_hours", 0))
        # mapear score a 5 clases (softmax sintético)
        logits = np.array([
            score + 2.5,  # ESI1
            score + 1.5,  # ESI2
            score + 0.5,  # ESI3
            -score + 0.5, # ESI4
            -score + 1.5, # ESI5
        ], dtype=float)
        exps = np.exp(logits - logits.max())
        proba = exps / exps.sum()
        return proba.reshape(1, -1)

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        if self.feature_list:
            ordered = [features.get(k, 0.0) for k in self.feature_list]
        else:
            ordered = [features.get(k, 0.0) for k in FeatureEngineer.FEATURE_ORDER]
        return np.asarray(ordered).reshape(1, -1)

    def explain_shap(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        try:
            if shap is None or self.model is None:
                raise RuntimeError("SHAP no disponible")
            X = self._features_to_array(features)
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X)  # para modelos multi-clase -> lista
            # tomar clase de mayor probabilidad
            proba = self.predict_proba(features).flatten()
            top_cls = int(np.argmax(proba))
            values = sv[top_cls].flatten()
            names = self.feature_list or FeatureEngineer.FEATURE_ORDER
            pairs = list(zip(names, X.flatten(), values))
        except Exception:
            # fallback: ranking por pesos heurísticos
            weights = {
                "chest_pain": 1.5, "dyspnea": 1.4, "radiation_left_arm": 1.2,
                "sweating": 0.8, "nausea": 0.7, "massive_bleeding": 1.6,
                "neuro_deficit": 1.6, "duration_hours": 0.3, "age": 0.2
            }
            names = FeatureEngineer.FEATURE_ORDER
            pairs = [(n, float(features.get(n, 0.0)), float(weights.get(n, 0.05))) for n in names]
        # top 5
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        return [
            {"feature": name, "value": float(val), "contrib": float(contrib)}
            for name, val, contrib in pairs_sorted
        ]


# ---------- Motor de Reglas de Salvaguarda ----------

class RuleEngine:
    @staticmethod
    def hard_overrides(features: Dict[str, float], answers: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
        """Devuelve (se_aplica, motivo_dict, esi_override)"""
        # hipotensión si sbp < 90
        sbp = features.get("sbp", 0)
        chest = features.get("chest_pain", 0) >= 1
        dys = features.get("dyspnea", 0) >= 1
        bleed = features.get("massive_bleeding", 0) >= 1
        neuro = features.get("neuro_deficit", 0) >= 1

        if (chest and dys and sbp and sbp < 90):
            return True, {"reason": "dolor torácico + disnea + hipotensión"}, 1
        if neuro:
            return True, {"reason": "alteración neurológica focal de inicio súbito"}, 1
        if bleed:
            return True, {"reason": "hemorragia masiva textual"}, 1
        return False, {}, 0


# ---------- Capa Generativa (LLM) con guardrails ----------

class GenerativeLayer:
    def __init__(self) -> None:
        self.prompt_template = (
            "Eres un asistente clínico. Dados el JSON de entrada de un triage y la predicción del modelo, "
            "devuelve un nivel sugerido, estudios iniciales (categorías generales, sin dosis), riesgos y un resumen HCE. "
            "No proporciones dosis farmacológicas específicas; no hagas diagnóstico definitivo; siempre incluye disclaimer.\n\n"
            "INPUT_JSON=\n{input_json}\n\n"
            "PREDICCION=\n{pred_json}\n\n"
            "Salida en JSON con claves: suggested_level, initial_studies, risks, hce_text, disclaimer."
        )
        self.prompt_hash = sha256(self.prompt_template)

    def generate(self, state: Dict[str, Any], triage_result: Dict[str, Any]) -> Dict[str, Any]:
        # Si se permite y hay API key, aquí podrías integrar OpenAI/otro LLM. Dejamos plantilla local segura.
        input_json = json.dumps(state, ensure_ascii=False)
        pred_json = json.dumps(triage_result, ensure_ascii=False)

        # Generador local basado en reglas/plantillas (sin fármacos ni dosis)
        level = f"ESI {triage_result['predicted_ESI']} ({triage_result['mapped_manchester']})"
        flags = state.get("extracted_entities", {}).get("flags", {})
        studies: List[str] = []
        risks: List[str] = []
        if flags.get("chest_pain"):
            studies += ["ECG", "Troponinas", "Monitoreo SatO2", "Rx tórax si persiste dolor"]
            risks += ["Posible isquemia miocárdica"]
        if flags.get("massive_bleeding"):
            studies += ["Laboratorio básico", "Grupo y factor", "Evaluación hemostática"]
            risks += ["Shock hipovolémico"]
        if flags.get("neuro_deficit"):
            studies += ["Evaluación neurológica urgente", "TC de cerebro sin contraste"]
            risks += ["ACV isquémico/hemorrágico"]
        if not studies:
            studies = ["Evaluación médica inicial", "Signos vitales seriados", "Laboratorio básico según presentación"]

        # Nota HCE breve
        patient = state.get("patient", {})
        age = patient.get("age")
        cc = state.get("chief_complaint")
        hce_text = f"Paciente de {age} años. Motivo: {cc}. Entrevista y análisis automatizado con IA. Nivel sugerido: {level}."

        out = {
            "suggested_level": level,
            "initial_studies": studies,
            "risks": risks or ["Riesgos a determinar según evolución"],
            "hce_text": redact_med_doses(hce_text),
            "disclaimer": DISCLAIMER_TEXT,
            "prompt_hash": self.prompt_hash,
        }
        return out


# ---------- Esquemas API ----------

class TriageRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Si no se envía, se genera")
    step: Optional[str] = Field(None, description="Identificador del paso actual")
    input: Optional[str] = Field(None, description="Texto libre o valor")
    context: Optional[Dict[str, Any]] = Field(None, description="Estado parcial opcional")
    user_id: Optional[str] = Field(None, description="Usuario que opera (para auditoría)")


class ExportFormat(str):
    pass


# ---------- App FastAPI y Middleware ----------

app = FastAPI(title="Triage Inteligente IA", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Componentes globales
NLP = ClinicalNLP()
FE = FeatureEngineer()
CLF = ESIClassifier()
RULES = RuleEngine()
GEN = GenerativeLayer()
DTM = DecisionTreeManager()


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    request_id = new_id("req")
    request.state.request_id = request_id
    response: Response = await call_next(request)
    # No escribimos aquí; el endpoint realizará el log con detalles de in/out
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Model-Version"] = MODEL_VERSION
    response.headers["X-Pipeline-Hash"] = PIPELINE_FINGERPRINT
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": MODEL_VERSION,
        "pipeline_hash": PIPELINE_FINGERPRINT,
        "db": os.path.abspath(DB_PATH),
    }
# === Configuración de la base de datos SQLite ===
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).with_name("triage_sessions.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            patient_json TEXT,
            chief_complaint TEXT,
            extracted_entities_json TEXT,
            derived_features_json TEXT,
            predicted_esi INTEGER,
            mapped_manchester TEXT,
            recommendation TEXT,
            override_applied INTEGER,
            override_reason TEXT,
            differentials_json TEXT,
            shap_top_json TEXT,
            llm_summary_json TEXT,
            disclaimer_ack INTEGER,
            state_json TEXT
        )
        """)
@app.on_event("startup")
async def startup_event():
    init_db()


# ---------- Lógica del endpoint /triage ----------

@app.post("/triage")
async def triage(req: TriageRequest, request: Request):
    request_id = request.state.request_id
    user_id = req.user_id

    # Cargar o crear sesión
    session_id = req.session_id or new_id("sess")
    state = store.load_state(session_id) or {
        "session_id": session_id,
        "patient": {},
        "chief_complaint": "",
        "timeline": [],
        "answers": {},
        "extracted_entities": {},
        "derived_features": {},
        "model_output": {},
        "rules_override": {},
        "final_triage": {},
        "shap": {},
        "llm_summary": {},
        "disclaimer_ack": False,
    }

    # Merge de contexto externo limitado (sin sobrescribir campos críticos)
    if req.context:
        state.update({k: v for k, v in req.context.items() if k in {"answers", "patient"}})

    step = req.step
    text = (req.input or "").strip()

    if not step:
        # Determinar siguiente paso
        step, question = DTM.next(state)
        store.save_state(session_id, state)
        out = {"session_id": session_id, "next_question": question, "completed": False, "state_snapshot": _public_state(state)}
        store.log_audit(request_id, user_id, session_id, req.model_dump(), out, "/triage")
        return out

    # Procesar input por step
    if step == DecisionTreeManager.IDENT:
        # Esperamos: nombre, edad, ciudad, ocupación (pseudonimizamos los identificadores)
        # Extraemos edad por regex
        age_m = re.search(r"(\d{1,3})\s*(anos|años|a$|años de edad)?", text.lower())
        age = safe_int(age_m.group(1)) if age_m else None
        state["patient"] = {
            "name_hash": pseudonymize(text),
            "age": age,
            "city_hash": pseudonymize(text),
            "occupation_hash": pseudonymize(text),
        }
        state["answers"][DecisionTreeManager.IDENT] = {"raw": pseudonymize(text)}

    elif step == DecisionTreeManager.CC:
        state["chief_complaint"] = text
        ents = NLP.extract(text)
        state["extracted_entities"] = ents
        state["answers"][DecisionTreeManager.CC] = {"text": text}

    elif step == DecisionTreeManager.ADAPT:
        # Guardamos texto de follow-up; volvemos a extraer entidades (enriquecer)
        ents_new = NLP.extract(text)
        ents = state.get("extracted_entities", {})
        # Merge simple de listas y flags OR
        def merge_flags(a: Dict[str, bool], b: Dict[str, bool]) -> Dict[str, bool]:
            keys = set(a.keys()) | set(b.keys())
            return {k: bool(a.get(k)) or bool(b.get(k)) for k in keys}
        merged = {
            "duration_hours": ents.get("duration_hours") or ents_new.get("duration_hours"),
            "symptoms": sorted(list(set(ents.get("symptoms", []) + ents_new.get("symptoms", [])))),
            "locations": sorted(list(set(ents.get("locations", []) + ents_new.get("locations", [])))),
            "drugs": sorted(list(set(ents.get("drugs", []) + ents_new.get("drugs", [])))),
            "conditions": sorted(list(set(ents.get("conditions", []) + ents_new.get("conditions", [])))),
            "flags": merge_flags(ents.get("flags", {}), ents_new.get("flags", {})),
        }
        state["extracted_entities"] = merged
        state["answers"][DecisionTreeManager.ADAPT] = {"text": text}

    elif step == DecisionTreeManager.MEDS:
        state["answers"][DecisionTreeManager.MEDS] = {"text": redact_med_doses(text)}

    elif step == DecisionTreeManager.PMH:
        state["answers"][DecisionTreeManager.PMH] = {"text": text}

    elif step == DecisionTreeManager.RISK:
        state["answers"][DecisionTreeManager.RISK] = {"text": text}

    elif step == DecisionTreeManager.CONSENT:
        state["disclaimer_ack"] = text.lower() in {"si", "sí", "acepto", "ok", "de acuerdo", "aceptar"}
        state["answers"][DecisionTreeManager.CONSENT] = {"text": text, "ack": state["disclaimer_ack"]}

    else:
        state["timeline"].append({"ts": now_utc(), "event": f"Unknown step input for {step}", "input": text})

    # timeline
    state["timeline"].append({"ts": now_utc(), "step": step, "input": text})

    # ¿Ya completamos?
    next_step, question = DTM.next(state)

    # Si ya completado (DONE), ejecutamos pipeline predictivo+generativo
    completed = (next_step == DecisionTreeManager.DONE)
    triage_result = None

    if completed:
        # features
        feats = FE.build(state)
        state["derived_features"] = feats
        # reglas de salvaguarda
        ov_applied, reason, esi_override = RULES.hard_overrides(feats, state.get("answers", {}))
        proba = CLF.predict_proba(feats).flatten()
        pred_idx = int(np.argmax(proba))
        predicted_ESI = ESI_CLASSES[pred_idx]
        if ov_applied:
            predicted_ESI = min(predicted_ESI, esi_override)  # override a más urgente
        mapped = ESI_TO_MANCHESTER.get(predicted_ESI)
        # diferenciales demo (en producción, usar modelo NLU)
        diffs = []
        flags = state.get("extracted_entities", {}).get("flags", {})
        if flags.get("chest_pain"):
            diffs = ["Síndrome coronario agudo", "Reflujo gastroesofágico", "Costocondritis"]
        elif flags.get("neuro_deficit"):
            diffs = ["ACV isquémico", "ACV hemorrágico", "Migraña complicada"]
        elif flags.get("massive_bleeding"):
            diffs = ["Trauma", "Trastorno hemostático", "Úlcera sangrante"]
        else:
            diffs = ["Etiología a precisar"]

        recommendation = ""
        if predicted_ESI in (1, 2):
            recommendation = "Derivar a emergencias para evaluación inmediata (ECG <10 min si dolor torácico)."
        elif predicted_ESI == 3:
            recommendation = "Atención urgente con evaluación clínica y estudios según presentación."
        else:
            recommendation = "Atención no inmediata; control clínico y signos de alarma."

        shap_top = CLF.explain_shap(feats)

        triage_result = {
            "predicted_ESI": predicted_ESI,
            "mapped_manchester": mapped,
            "possible_differentials": diffs,
            "recommendation": recommendation,
            "proba": {str(k): float(v) for k, v in zip(ESI_CLASSES, proba)},
            "shap_top": shap_top,
            "override_applied": bool(ov_applied),
            "override_reason": reason,
        }
        state["model_output"] = triage_result
        state["rules_override"] = {"applied": bool(ov_applied), **reason}

        # capa generativa con guardrails
        gen = GEN.generate(state, triage_result)
        state["llm_summary"] = gen

        state["final_triage"] = {
            "esi": predicted_ESI,
            "manchester": mapped,
            "recommendation": recommendation,
        }

    # Guardar estado y preparar respuesta
    store.save_state(session_id, state)

    if completed:
        reply = {
            "session_id": session_id,
            "completed": True,
            "triage_result": triage_result,
            "generative_summary": state.get("llm_summary"),
        }
    else:
        reply = {
            "session_id": session_id,
            "next_question": question,
            "completed": False,
            "state_snapshot": _public_state(state),
        }

    store.log_audit(request_id, user_id, session_id, req.model_dump(), reply, "/triage")
    return reply


# ---------- Export ----------

@app.get("/export/{session_id}")
async def export_session(session_id: str, format: Optional[str] = None):
    state = store.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    fmt = (format or "json").lower()
    if fmt == "json":
        return Response(
            content=json.dumps(_public_state(state), ensure_ascii=False, indent=2),
            media_type="application/json",
        )
    if fmt == "csv":
        # aplanar features + resultado
        row: Dict[str, Any] = {
            "session_id": session_id,
            "age": state.get("patient", {}).get("age"),
        }
        row.update(state.get("derived_features", {}))
        mo = state.get("model_output", {})
        for k in ["predicted_ESI", "mapped_manchester", "override_applied"]:
            row[k] = mo.get(k)
        df = pd.DataFrame([row])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return Response(content=csv_bytes, media_type="text/csv")

    raise HTTPException(status_code=400, detail="Formato no soportado (use json|csv)")


# ---------- Utilidades privadas ----------

def _public_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Oculta PII y retorna un snapshot apto para UI/API."""
    keep = {
        "session_id": state.get("session_id"),
        "patient": {"age": state.get("patient", {}).get("age")},
        "chief_complaint": state.get("chief_complaint"),
        "answers": {
            k: (v if k != DecisionTreeManager.IDENT else {"raw": "[HASHED]"})
            for k, v in state.get("answers", {}).items()
        },
        "extracted_entities": state.get("extracted_entities", {}),
        "derived_features": state.get("derived_features", {}),
        "model_output": state.get("model_output", {}),
        "rules_override": state.get("rules_override", {}),
        "final_triage": state.get("final_triage", {}),
        "shap": state.get("shap", {}),
        "llm_summary": state.get("llm_summary", {}),
        "disclaimer_ack": bool(state.get("disclaimer_ack")),
    }
    return keep


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("UVICORN_PORT", 8000)))

