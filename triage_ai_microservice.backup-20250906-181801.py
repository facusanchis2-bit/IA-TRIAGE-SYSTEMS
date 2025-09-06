from __future__ import annotations

import hashlib, json, os, re, uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)  # <- crea la carpeta si falta
log_handler = RotatingFileHandler(os.path.join(LOG_DIR, "triage.log"), maxBytes=2_000_000, backupCount=5)

from logging.handlers import RotatingFileHandler

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ================== META / CONFIG ==================
APP_VERSION = "3.0.0-P3-postgres-sqlalchemy"
BASE_DIR = Path(__file__).parent

# DB URL: si no está seteada, fallback a SQLite (archivo en el proyecto)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR/'triage_sessions.db'}")

SESSIONS: Dict[str, Dict[str, Any]] = {}

# Auth básica para /admin y export
security = HTTPBasic()
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")

# Sal para pseudonimizar PHI (CAMBIAR en producción)
PHI_SALT = os.getenv("PHI_SALT", "change-me-long-random-salt")

# ================== DB (SQLAlchemy) ==================
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR/'triage_sessions.db'}")

engine: Engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=_int_env("DB_POOL_SIZE", 5),
    max_overflow=_int_env("DB_MAX_OVERFLOW", 5),
    pool_recycle=_int_env("DB_POOL_RECYCLE", 1800),  # seg.
    pool_timeout=_int_env("DB_POOL_TIMEOUT", 10),    # seg.
)

def init_db():
    """
    Crea la tabla sessions si no existe y asegura columnas clave.
    Funciona en Postgres y SQLite con SQL estándar.
    """
    with engine.begin() as con:
        # Crear tabla si no existe
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            patient_json TEXT,
            patient_name TEXT,
            patient_hash TEXT,
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
        """))

        # En SQLite no existe fácil "ADD COLUMN IF NOT EXISTS", en Postgres sí.
        # Para portabilidad, chequeamos esquema por PRAGMA/INFORMATION_SCHEMA simplificado:
        # Aquí optamos por intentar ADD COLUMN y capturar errores benignos.
        for col, typ in [
            ("patient_name","TEXT"),
            ("patient_hash","TEXT"),
            ("chief_complaint","TEXT"),
            ("override_reason","TEXT"),
        ]:
            try:
                con.execute(text(f"ALTER TABLE sessions ADD COLUMN {col} {typ}"))
            except Exception:
                # Ya existe
                pass

# ================== TEXTO / UI ==================
UI_QUESTION = {
    "nombre_apellido": "¿Cuál es tu nombre y apellido?",
    "edad": "¿Qué edad tenés? (en años)",
    "domicilio": "¿En qué ciudad o localidad vivís?",
    "ocupacion": "¿Trabajás? ¿En qué?",
    "motivo": "Contanos con tus palabras, ¿cuál es el motivo principal de consulta?",
    "fc": "Si podés, decinos tu frecuencia cardíaca (latidos por minuto). Si no sabés, dejalo vacío.",
    "pas": "Si mediste la presión: ¿cuál fue la sistólica (número grande, mmHg)? Si no sabés, dejalo vacío.",
    "sat02": "¿Cuál es tu saturación de oxígeno (SatO₂ %), si la mediste? Si no sabés, dejalo vacío.",
    "temp": "¿Tenés temperatura corporal medida? (en °C). Si no, dejalo vacío.",
    "antecedentes": "¿Tenés enfermedades previas? (presión alta, diabetes, etc.). Si ninguna, escribí 'ninguna'.",
    "medicacion": "¿Qué medicaciones tomás actualmente? Si ninguna, escribí 'ninguna'.",
    "alergias": "¿Alergia a medicamentos o alimentos? Si ninguna, escribí 'no'.",
    "factores_riesgo": "¿Factores de riesgo? (presión alta, diabetes, colesterol, fumás, antecedentes familiares).",
    "sintomas_asoc": "¿Qué otras cosas notás? (fiebre, tos, falta de aire, latidos muy fuertes, desmayo, vómitos…).",
    "alivio": "¿Hay algo que lo alivie? (reposo, alguna medicación, postura).",
    "desencadenantes": "¿Qué lo desencadena o empeora? (esfuerzo, comidas, estrés, movimiento).",
    "caracter": "¿Cómo lo describís? (opresivo, punzante, quemante, cólico, etc.)",
    "intensidad": "En una escala de 0 a 10, ¿qué intensidad tiene?",
    "localizacion": "¿Dónde lo sentís principalmente? (pecho, panza, cabeza, espalda, garganta, al orinar, golpe/trauma…).",
    "consentimiento": "Escribí 'sí' para confirmar que leíste la recomendación asistida (no reemplaza la evaluación médica).",
}
LABELS = {
    "nombre_apellido":"nombre y apellido","edad":"edad","domicilio":"domicilio","ocupacion":"ocupación",
    "motivo":"motivo de consulta","fc":"frecuencia cardíaca","pas":"presión sistólica","sat02":"saturación O₂","temp":"temperatura (°C)",
    "antecedentes":"enfermedades previas","medicacion":"medicación","alergias":"alergias","factores_riesgo":"factores de riesgo",
    "sintomas_asoc":"síntomas asociados","alivio":"qué lo alivia","desencadenantes":"qué lo empeora","caracter":"cómo lo describís",
    "intensidad":"intensidad (0–10)","localizacion":"dónde lo sentís","dirigidas":"preguntas dirigidas","consentimiento":"consentimiento"
}
MANCHESTER_MAP = {1:"Red",2:"Orange",3:"Yellow",4:"Green",5:"Blue"}

# ================== FASTAPI / CORS ==================
# Configuración de logging rotativo
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

log_handler = RotatingFileHandler(
    "logs/triage.log",   # archivo donde se guardan los logs
    maxBytes=5*1024*1024, # 5 MB por archivo
    backupCount=5         # mantiene hasta 5 archivos de backup
)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

# Logger raíz
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# También seguimos viendo logs en consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

app = FastAPI(title="Triage IA")

# CORS: permite que el frontend (Nginx) hable con la API
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
app.include_router(admin_router)

)

@app.on_event("startup")
async def _startup():
    init_db()

# ================== UTILIDADES ==================
def normalize(s:str)->str: return (s or "").lower().strip()
def now_iso()->str: return datetime.utcnow().isoformat(timespec="seconds")

def gen_session_id()->str:
    raw=f"{datetime.utcnow().isoformat()}-{APP_VERSION}-{uuid.uuid4().hex}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def patient_hash(name:str)->str:
    txt = (name or "").strip().lower().encode("utf-8")
    return hashlib.sha256((PHI_SALT.encode("utf-8") + b"|" + txt)).hexdigest()

def append_timeline(st,ev,data=None): st.setdefault("timeline",[]).append({"ts":now_iso(),"event":ev,"data":data})

# ================== CLASIFICACIÓN / REGLAS ==================
def classify_pathway(text:str)->str:
    t=normalize(text)
    if any(k in t for k in ["pecho","precordial","torac"]): return "pecho"
    if any(k in t for k in ["panza","vientre","abdomen","estómago","estomago"]): return "abdomen"
    if any(k in t for k in ["cabeza","cefalea","migra"]): return "cabeza"
    if any(k in t for k in ["tos","respirar","falta de aire","ahogo","garganta","silbidos"]): return "respiratorio"
    if any(k in t for k in ["orinar","pis","ardor","orina","urin"]): return "urinario"
    if any(k in t for k in ["golpe","trauma","caída","fractura","esguince","torcedura"]): return "trauma"
    if any(k in t for k in ["espalda","lumbalgia","lumba","lomo","ciática","ciatica"]): return "lumbalgia"
    if any(k in t for k in ["piel","erup","manchas","ronchas","sarpul","rash"]): return "piel"
    if any(k in t for k in ["mareo","vértigo","vertigo","inestabilidad"]): return "mareo"
    return "generico"

def followups_for_path(path:str)->Optional[str]:
    m = {
        "pecho": "Para el dolor en el pecho: ¿cuándo empezó? ¿Se va al brazo izquierdo o mandíbula? ¿Te falta el aire? ¿Sudor frío? ¿Náuseas?",
        "abdomen": "Para dolor de panza: ¿en qué zona? ¿Fiebre? ¿Vómitos? ¿Diarrea o estreñimiento? ¿Embarazo posible?",
        "cabeza": "Dolor de cabeza: ¿empezó de golpe? ¿Rigidez de cuello? ¿Dificultad para hablar o mover? ¿Molesta la luz?",
        "respiratorio": "Respiratorio: ¿tos con flema? ¿Fiebre? ¿Te falta el aire al caminar o en reposo? ¿Dolor al respirar?",
        "urinario": "Al orinar: ¿ardor? ¿Aumentaron las ganas? ¿Sangre en la orina? ¿Dolor en cintura?",
        "trauma": "Trauma: ¿cómo fue el golpe? ¿Podés apoyar o mover? ¿Deformidad? ¿Dolor con el movimiento? ¿Pérdida de conocimiento?",
        "lumbalgia": "Dolor lumbar: ¿desde cuándo? ¿Se va a la pierna? ¿Hormigueo/debilidad? ¿Incontinencia? ¿Fiebre o pérdida de peso?",
        "piel": "Piel/erupción: ¿desde cuándo? ¿Pica? ¿Se extiende? ¿Fiebre? ¿Con dolor o ampollas?",
        "mareo": "Mareo/vértigo: ¿súbito o gradual? ¿Empeora con movimientos? ¿Zumbidos, vómitos, visión doble, debilidad o dificultad para hablar?",
    }
    return m.get(path)

def build_steps(path:str)->List[str]:
    return [
        "nombre_apellido","edad","domicilio","ocupacion",
        "motivo",
        "fc","pas","sat02","temp",
        "antecedentes","medicacion","alergias","factores_riesgo",
        "sintomas_asoc","alivio","desencadenantes","caracter","intensidad","localizacion",
        "dirigidas",
        "consentimiento",
    ]

_WEIGHTS={ "chest_pain":2.5,"dyspnea":2.0,"radiation_left_arm":1.5,"diaphoresis":1.0,"nausea":0.5,"sudden_onset":1.0,
           "palpitations":0.5,"syncope":1.5,"diabetes":0.5,"htn":0.5 }

def simple_ner_and_features(ans:Dict[str,str])->Dict[str,Any]:
    fields = ["motivo","sintomas_asoc","caracter","localizacion","desencadenantes","alivio","antecedentes","factores_riesgo","dirigidas"]
    txt = " ".join(normalize(ans.get(k,"")) for k in fields)
    f={
        "chest_pain": int(any(k in txt for k in ["pecho","precordial"])),
        "dyspnea": int(any(k in txt for k in ["falta de aire","ahogo"])),
        "palpitations": int(any(k in txt for k in ["latidos fuertes","palpitaciones"])),
        "syncope": int(any(k in txt for k in ["desmayo","se desmayó","sincope"])),
        "radiation_left_arm": int(any(k in txt for k in ["brazo izquierdo","mandíbula"])),
        "diaphoresis": int(any(k in txt for k in ["sudor","transpiración fría"])),
        "nausea": int(any(k in txt for k in ["náusea","nausea","vómit","vomit"])),
        "sudden_onset": int(any(k in txt for k in ["inicio súbito","de repente","brusco"])),
        "altered_consciousness": int(any(k in txt for k in ["inconsciente","no responde","confuso","somnoliento"])),
        "focal_deficit_sudden": int(any(k in txt for k in ["cara caída","hablar mal","debilidad del brazo","no mueve un lado","dificultad para hablar"])),
        "massive_bleeding": int(any(k in txt for k in ["sangrado abundante","hemorragia masiva"])),
        "hypotension": int("presión muy baja" in txt or "hipotens" in txt),
        "hemodynamic_instability": int("shock" in txt or "muy pálido" in txt),
        "diabetes": int("diabetes" in txt),
        "htn": int("presión alta" in txt or "hipertens" in txt),
    }
    def fnum(key, default=0.0):
        v = str(ans.get(key,"")).strip()
        if v == "": return default
        try: return float(v.replace(",","."))
        except: return default
    f["severity_num"]=max(0.0,min(10.0,fnum("intensidad",0.0)))
    f["hr"]=fnum("fc",0.0); f["sbp"]=fnum("pas",0.0); f["spo2"]=fnum("sat02",0.0); f["temp"]=fnum("temp",0.0)
    entities = {
        "vitals":{"hr":f["hr"],"sbp":f["sbp"],"spo2":f["spo2"],"temp":f["temp"]},
        "comorbidities":[k for k in ["diabetes","hipertensión","presión alta","asma","epoc","cardiopatía"] if k in txt],
        "phrases":{"sudden_onset":bool(f["sudden_onset"])}
    }
    return {"features":f,"entities":entities}

def hard_overrides(f:Dict[str,Any])->Optional[Dict[str,Any]]:
    if f.get("massive_bleeding"): return {"esi":1,"reason":"Sangrado abundante"}
    if f.get("altered_consciousness") and f.get("focal_deficit_sudden"):
        return {"esi":1,"reason":"Alteración de conciencia + signos focales (ACV)"}
    if f.get("chest_pain") and f.get("dyspnea") and (f.get("hypotension") or f.get("hemodynamic_instability")):
        return {"esi":1,"reason":"Dolor de pecho + falta de aire + inestabilidad"}
    if f.get("spo2",0) and f["spo2"] < 90: return {"esi":2,"reason":"SatO₂ < 90%"}
    if f.get("sbp",0) and f["sbp"] < 90: return {"esi":2,"reason":"PAS < 90 mmHg"}
    if (f.get("temp",0)>=39.5) and (f.get("hr",0)>=120) and f.get("hypotension",0)==1:
        return {"esi":2,"reason":"Riesgo de sepsis (demo)"}
    return None

def risk_score(f:Dict[str,Any])->float:
    s=sum(_WEIGHTS[k] for k in _WEIGHTS if f.get(k)) + 0.1*float(f.get("severity_num",0))
    if f.get("spo2",0) and f["spo2"]<93: s += 0.7
    if f.get("temp",0) and f["temp"]>=39.5: s += 0.6
    if f.get("sbp",0) and f["sbp"]<100: s += 0.4
    if f.get("hr",0) and f["hr"]>=120: s += 0.4
    if f.get("sudden_onset"): s += 0.6
    return s

_THRESHOLDS = {2:6.5, 3:3.5, 4:1.5}
def predict_esi(f:Dict[str,Any])->Tuple[int,Optional[Dict[str,Any]],List[Dict[str,Any]],Dict[str,Any]]:
    ov=hard_overrides(f)
    if ov:
        esi=max(1,min(2,ov["esi"]))
        shap=[{"feature":ov["reason"],"value":1,"contrib":6.0}]
        return esi,ov,shap,{"low_confidence":False,"note":"Override clínico"}
    s=risk_score(f)
    if   s>=_THRESHOLDS[2]: esi=2; margin=s-_THRESHOLDS[2]
    elif s>=_THRESHOLDS[3]: esi=3; margin=s-_THRESHOLDS[3]
    elif s>=_THRESHOLDS[4]: esi=4; margin=s-_THRESHOLDS[4]
    else:                   esi=5; margin=(_THRESHOLDS[4]-s)
    low_conf = abs(margin) < 0.3
    if low_conf and esi in (3,4):  # prudencia
        esi -= 1
    shap=[{"feature":k,"value":(f[k] if not isinstance(f[k],bool) else 1),"contrib":(_WEIGHTS.get(k,0) or (0.1 if k=="severity_num" else 0))}
          for k in f if f.get(k)]
    for vk,contrib in [("spo2",0.9),("sbp",0.6),("hr",0.4),("temp",0.5)]:
        if f.get(vk,0): shap.append({"feature":vk,"value":f[vk],"contrib":contrib})
    shap.sort(key=lambda x:-x["contrib"])
    return esi,None,shap[:10],{"low_confidence":low_conf, "note":"borde umbral" if low_conf else "", "score":round(s,2)}

def llm_generate_summary(triage:Dict[str,Any], state:Dict[str,Any])->Dict[str,Any]:
    esi=triage.get("predicted_ESI")
    if esi in (1,2):
        studies=["ECG","SatO2 continua","Troponinas","Monitoreo"]; risks=["Riesgo cardiovascular/respiratorio agudo"]
    elif esi==3:
        studies=["Laboratorio básico","ECG si dolor en pecho","Rx tórax si falta de aire"]; risks=["Evaluación en guardia"]
    else:
        studies=["Control ambulatorio"]; risks=["Bajo riesgo aparente"]
    hce=f"Paciente: {state.get('patient',{}).get('nombre_apellido','')} | Motivo: {state.get('answers',{}).get('motivo','')}. ESI {esi} ({triage.get('mapped_manchester')})."
    return {"suggested_level":f"ESI {esi} ({triage.get('mapped_manchester')})","initial_studies":studies,"risks":risks,
            "hce_text":hce,"disclaimer":"Recomendación asistida por IA; no reemplaza evaluación médica. Sin dosis."}

# ================== UI helpers ==================
def ui_meta(state:Dict[str,Any])->Dict[str,Any]:
    step = state["steps"][state["idx"]]
    field_type="text"; placeholder="Escribí tu respuesta"
    if step in ("edad","intensidad","fc","pas","sat02","temp"):
        field_type="number"; placeholder="Ingresar número"
    return {"current_step":step,"current_label":LABELS.get(step,step),"step_index":state["idx"]+1,
            "step_total":len(state["steps"]),"field_type":field_type,"placeholder":placeholder,
            "title":"Ayudanos a orientarte",
            "subtitle":"Te vamos a hacer algunas preguntas rápidas para estimar la urgencia."}

def next_question_for(state:Dict[str,Any])->str:
    step=state["steps"][state["idx"]]
    if step=="dirigidas":
        path = state.get("path","generico")
        return followups_for_path(path) or "Contanos algo más que creas importante."
    return UI_QUESTION.get(step,"Continuar…")

# ================== API ==================
@app.get("/health")
def health(): return {"status":"ok","version":APP_VERSION}

@app.post("/triage")
async def triage(request:Request):
    try: body=await request.json()
    except: body={}
    session_id=(body.get("session_id") or "").strip()
    user_input=(body.get("input") or "").strip()

    # Crear sesión
    if not session_id:
        session_id=gen_session_id()
        steps=build_steps("generico")
        state={"session_id":session_id,"created_at":now_iso(),"answers":{},"timeline":[],"idx":0,
               "steps":steps,"path":"generico","version":APP_VERSION,"disclaimer_ack":False}
        SESSIONS[session_id]=state
        append_timeline(state,"session_started")
        return {"session_id":session_id,"next_question":next_question_for(state),"completed":False,"ui":ui_meta(state)}

    # Recuperar estado
    state=SESSIONS.get(session_id)
    if not state: raise HTTPException(404,"Sesión no encontrada")

    step=state["steps"][state["idx"]]
    if not user_input:
        defaults={"alergias":"no","antecedentes":"ninguna","medicacion":"ninguna","ocupacion":"no"}
        user_input=defaults.get(step,"")

    # Guardar respuesta
    ans=state["answers"]; ans[step]=user_input; append_timeline(state,"answer",{"step":step,"input":user_input})

    # Datos paciente + hash
    name = (ans.get("nombre_apellido") or "").strip()
    state["patient"]={
        "nombre_apellido": name,
        "edad": ans.get("edad"),
        "domicilio": ans.get("domicilio"),
        "ocupacion": ans.get("ocupacion")
    }
    state["patient_name"] = name
    state["patient_hash"] = patient_hash(name)
    state["chief_complaint"]= (ans.get("motivo","") or "").strip()

    # Rama desde MOTIVO
    if step=="motivo":
        state["path"]=classify_pathway(user_input)
        state["steps"]=build_steps(state["path"])
        state["idx"]=state["steps"].index("motivo")  # reposicionar

    # Afinar rama con localización si aún genérico
    if step=="localizacion" and state.get("path","generico")=="generico":
        new_path=classify_pathway(user_input)
        if new_path!="generico": state["path"]=new_path

    # ¿Completó? (Requiere consentimiento explícito)
    if step=="consentimiento":
        if normalize(user_input) not in ("si","sí"):
            return {"session_id":session_id,"completed":False,"ui":ui_meta(state),
                    "next_question":"Para continuar, escribí 'sí' para confirmar el consentimiento informado."}

        nlp=simple_ner_and_features(ans)
        features, entities = nlp["features"], nlp["entities"]
        esi, override, shap_top, meta = predict_esi(features)

        if esi==1:
            diffs=["Emergencia mayor"]; rec="Atención inmediata. Monitoreo y equipo de emergencias."
        elif esi==2:
            diffs=["Posible evento serio (cardio/respiratorio)"]; rec="Urgencias (ECG, Troponinas, monitoreo)."
        elif esi==3:
            diffs=["Evaluación en guardia"]; rec="Evaluación en guardia."
        elif esi==4:
            diffs=["Cuadro leve/moderado"]; rec="Atención diferida/ambulatoria."
        else:
            diffs=["Consulta simple"]; rec="Alta con recomendaciones."

        triage_result={
            "predicted_ESI":esi,
            "mapped_manchester":MANCHESTER_MAP.get(esi),
            "possible_differentials":diffs,
            "recommendation":rec,
            "shap_top":shap_top,
            "override_applied":bool(override),
            "override_reason":override,
            "uncertainty": meta
        }
        gen=llm_generate_summary(triage_result,state)

        state["extracted_entities"]=entities
        state["derived_features"]=features
        state["final_triage"]=triage_result
        state["llm_summary"]=gen
        state["disclaimer_ack"]=True

        save_session_to_db(state)
        append_timeline(state,"completed",{"esi":esi})

        return {"session_id":session_id,"completed":True,"triage_result":triage_result,"generative_summary":gen}

    # Avanzar
    state["idx"] += 1
    if state["steps"][state["idx"]]=="dirigidas" and not followups_for_path(state.get("path","generico")):
        state["idx"] += 1
    return {"session_id":session_id,"next_question":next_question_for(state),"completed":False,"ui":ui_meta(state)}

def save_session_to_db(state: Dict[str, Any]) -> None:
    triage = state.get("final_triage", {})
    gen = state.get("llm_summary", {})
    with engine.begin() as con:
        con.execute(text("""
        INSERT INTO sessions(
          session_id, created_at, patient_json, patient_name, patient_hash,
          chief_complaint, extracted_entities_json, derived_features_json,
          predicted_esi, mapped_manchester, recommendation,
          override_applied, override_reason, differentials_json,
          shap_top_json, llm_summary_json, disclaimer_ack, state_json
        ) VALUES (
          :session_id, :created_at, :patient_json, :patient_name, :patient_hash,
          :chief_complaint, :extracted_entities_json, :derived_features_json,
          :predicted_esi, :mapped_manchester, :recommendation,
          :override_applied, :override_reason, :differentials_json,
          :shap_top_json, :llm_summary_json, :disclaimer_ack, :state_json
        )
        ON CONFLICT(session_id) DO UPDATE SET
          created_at=excluded.created_at,
          patient_json=excluded.patient_json,
          patient_name=excluded.patient_name,
          patient_hash=excluded.patient_hash,
          chief_complaint=excluded.chief_complaint,
          extracted_entities_json=excluded.extracted_entities_json,
          derived_features_json=excluded.derived_features_json,
          predicted_esi=excluded.predicted_esi,
          mapped_manchester=excluded.mapped_manchester,
          recommendation=excluded.recommendation,
          override_applied=excluded.override_applied,
          override_reason=excluded.override_reason,
          differentials_json=excluded.differentials_json,
          shap_top_json=excluded.shap_top_json,
          llm_summary_json=excluded.llm_summary_json,
          disclaimer_ack=excluded.disclaimer_ack,
          state_json=excluded.state_json
        """), {
            "session_id": state["session_id"],
            "created_at": state["created_at"],
            "patient_json": json.dumps(state.get("patient",{}), ensure_ascii=False),
            "patient_name": state.get("patient_name",""),
            "patient_hash": state.get("patient_hash",""),
            "chief_complaint": state.get("chief_complaint",""),
            "extracted_entities_json": json.dumps(state.get("extracted_entities",{}), ensure_ascii=False),
            "derived_features_json": json.dumps(state.get("derived_features",{}), ensure_ascii=False),
            "predicted_esi": triage.get("predicted_ESI"),
            "mapped_manchester": triage.get("mapped_manchester"),
            "recommendation": triage.get("recommendation"),
            "override_applied": 1 if triage.get("override_applied") else 0,
            "override_reason": (triage.get("override_reason") or {}).get("reason") if isinstance(triage.get("override_reason"),dict) else triage.get("override_reason"),
            "differentials_json": json.dumps(triage.get("possible_differentials",[]), ensure_ascii=False),
            "shap_top_json": json.dumps(triage.get("shap_top",[]), ensure_ascii=False),
            "llm_summary_json": json.dumps(gen, ensure_ascii=False),
            "disclaimer_ack": 1 if state.get("disclaimer_ack") else 0,
            "state_json": json.dumps(state, ensure_ascii=False),
        })

# ================== PANEL + EXPORT (protegidos) ==================
def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok = (credentials.username == ADMIN_USER and credentials.password == ADMIN_PASS)
    if not ok:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True
@app.get("/admin/sessions")
def list_sessions(
    auth: bool = Depends(require_admin),
    esi: Optional[int] = Query(None),
    q: Optional[str] = Query(None, min_length=0),
    name: Optional[str] = Query(None, min_length=0),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    order_by: str = Query("created_at_desc"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    where = []
    args: Dict[str, Any] = {}
    if esi in (1,2,3,4,5): where.append("predicted_esi = :esi"); args["esi"]=esi
    if q:
        where.append("(LOWER(COALESCE(chief_complaint,'')) LIKE LOWER(:q) OR LOWER(COALESCE(patient_name,'')) LIKE LOWER(:q))")
        args["q"]=f"%{q}%"
    if name:
        where.append("LOWER(COALESCE(patient_name,'')) LIKE LOWER(:name)")
        args["name"]=f"%{name}%"
    if from_: where.append("created_at >= :from_"); args["from_"]=from_
    if to:    where.append("created_at <= :to");    args["to"]=to
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    order_sql = "created_at DESC" if order_by.lower()=="created_at_desc" else "created_at ASC"

    with engine.begin() as con:
        total = con.execute(text(f"SELECT COUNT(*) FROM sessions {clause}"), args).scalar_one()
        rows = con.execute(text(f"""
            SELECT created_at, patient_name, patient_hash, chief_complaint,
                   predicted_esi, mapped_manchester, recommendation,
                   override_applied, override_reason, session_id
            FROM sessions
            {clause}
            ORDER BY {order_sql}
            LIMIT :limit OFFSET :offset
        """), {**args, "limit":limit, "offset":offset}).mappings().all()
    return JSONResponse({"total": total, "items": [dict(r) for r in rows]})

    # Filtros portables (SQLite/Postgres)
    where = []
    args: Dict[str, Any] = {}

    if esi in (1, 2, 3, 4, 5):
        where.append("predicted_esi = :esi"); args["esi"] = esi

    if q:
        where.append("(LOWER(COALESCE(chief_complaint,'')) LIKE LOWER(:q) OR LOWER(COALESCE(patient_name,'')) LIKE LOWER(:q))")
        args["q"] = f"%{q}%"

    if name:
        where.append("LOWER(COALESCE(patient_name,'')) LIKE LOWER(:name)")
        args["name"] = f"%{name}%"

    if from_:
        where.append("created_at >= :from_"); args["from_"] = from_
    if to:
        where.append("created_at <= :to"); args["to"] = to

    clause = ("WHERE " + " AND ".join(where)) if where else ""

    order_sql = "created_at DESC" if order_by.lower() == "created_at_desc" else "created_at ASC"

    with engine.begin() as con:
        # total
        total = con.execute(text(f"SELECT COUNT(*) AS c FROM sessions {clause}"), args).scalar_one()
        # page
        rows = con.execute(text(f"""
          SELECT created_at, patient_name, patient_hash, chief_complaint,
                 predicted_esi, mapped_manchester, recommendation,
                 override_applied, override_reason, session_id
          FROM sessions
          {clause}
          ORDER BY {order_sql}
          LIMIT :limit OFFSET :offset
        """), {**args, "limit": limit, "offset": offset}).mappings().all()
        return JSONResponse({"total": total, "items": [dict(r) for r in rows]})

@app.get("/admin/sessions/{session_id}")
def session_detail(session_id:str, auth: bool = Depends(require_admin)):
    with engine.begin() as con:
        r = con.execute(text("SELECT * FROM sessions WHERE session_id=:id"), {"id":session_id}).mappings().first()
        if not r: raise HTTPException(404, "not found")
        d=dict(r)
        for k in ["patient_json","extracted_entities_json","derived_features_json","differentials_json","shap_top_json","llm_summary_json","state_json"]:
            if d.get(k):
                try: d[k]=json.loads(d[k])
                except: pass
        return JSONResponse(d)

@app.get("/export/excel")
def export_excel(auth: bool = Depends(require_admin)):
    with engine.begin() as con:
        df = pd.read_sql(text("""
          SELECT created_at as Fecha, patient_name as Paciente, patient_hash as PacienteHash,
                 chief_complaint as Motivo, predicted_esi as ESI, mapped_manchester as Manchester,
                 recommendation as Recomendacion, override_applied as Override
          FROM sessions ORDER BY created_at DESC
        """), con)
        buf=BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w,index=False,sheet_name="Resumen")
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="triage_export.xlsx"'})

@app.get("/admin", response_class=HTMLResponse)
def admin_page(auth: bool = Depends(require_admin)):
    html="""
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panel Médico</title>
<style>
 body{font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}
 table{border-collapse:collapse;width:100%;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06)}
 th,td{border-bottom:1px solid #e2e8f0;padding:10px 12px;font-size:14px} th{background:#f1f5f9;text-align:left}
 .tag{padding:2px 8px;border-radius:9999px;font-size:12px}
 .Red{background:#fee2e2;color:#991b1b}.Orange{background:#ffedd5;color:#9a3412}
 .Yellow{background:#fef9c3;color:#854d0e}.Green{background:#dcfce7;color:#166534}
 .Blue{background:#dbeafe;color:#1e40af}
 .top{display:flex;gap:12px;align-items:center;margin-bottom:12px}
 .btn{padding:8px 12px;border-radius:10px;background:#0f172a;color:#fff;text-decoration:none}
 input,select{padding:8px;border:1px solid #cbd5e1;border-radius:10px}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 small{color:#475569}
</style>
<div class="top">
 <h1 style="margin:0;font-size:20px">Panel Médico</h1>
 <a class="btn" href="/export/excel">Descargar Excel</a>
</div>
<div class="row" style="margin:12px 0;">
 <input id="name" placeholder="Nombre del paciente..." style="min-width:220px">
 <input id="q" placeholder="Buscar por motivo o nombre..." style="min-width:260px">
 <select id="esi">
   <option value="">ESI (todos)</option>
   <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
 </select>
 <input id="from" type="datetime-local">
 <input id="to" type="datetime-local">
 <button class="btn" id="aplicar">Aplicar</button>
</div>
<small>Mostrando: Fecha, Paciente (click abre detalle), Motivo, ESI, Manchester, Recomendación, Override</small>
<table id="t"><thead><tr>
<th>Fecha</th><th>Paciente</th><th>Motivo</th><th>ESI</th><th>Manchester</th><th>Recomendación</th><th>Override</th>
</tr></thead><tbody></tbody></table>
<div class="row" style="margin-top:10px; justify-content:flex-end; gap:8px;">
  <button class="btn" id="prev">« Anterior</button>
  <span id="pager"></span>
  <button class="btn" id="next">Siguiente »</button>
</div>
<script>
let page = 0, pageSize = 50;

async function load(){
  const q = document.getElementById('q').value.trim();
  const name = document.getElementById('name').value.trim();
  const esi = document.getElementById('esi').value.trim();
  const from = document.getElementById('from').value.replace('T',' ');
  const to = document.getElementById('to').value.replace('T',' ');

  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (name) params.set('name', name);
  if (esi) params.set('esi', esi);
  if (from) params.set('from', from);
  if (to) params.set('to', to);
  params.set('order_by','created_at_desc');
  params.set('limit', pageSize.toString());
  params.set('offset', (page*pageSize).toString());

  const r = await fetch('/admin/sessions?' + params.toString(), {headers:{'Accept':'application/json'}});
  const data = await r.json();
  const rows = data.items || data; // compat con versión sin paginado

  const tb = document.querySelector('#t tbody');
  tb.innerHTML='';

  for (const x of rows){
    const tr=document.createElement('tr');
    const link = `/admin/sessions/${x.session_id}`;
    tr.innerHTML = `
      <td>${x.created_at||''}</td>
      <td><a href="${link}" target="_blank">${x.patient_name || '(sin nombre)'}</a><br><small>${x.patient_hash||''}</small></td>
      <td>${(x.chief_complaint||'').slice(0,120)}</td>
      <td>${x.predicted_esi ?? ''}</td>
      <td><span class="tag ${x.mapped_manchester||''}">${x.mapped_manchester||''}</span></td>
      <td>${(x.recommendation||'').slice(0,120)}</td>
      <td>${x.override_applied ? 'Sí' : 'No'}</td>`;
    tb.appendChild(tr);
  }

  // Paginador
  const total = data.total ?? rows.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  document.getElementById('pager').textContent = `Página ${page+1} / ${totalPages}`;
  document.getElementById('prev').disabled = (page <= 0);
  document.getElementById('next').disabled = (page+1 >= totalPages && data.total !== undefined);
}

document.getElementById('aplicar').onclick = () => { page = 0; load(); };
document.getElementById('prev').onclick = () => { if(page>0){ page--; load(); }};
document.getElementById('next').onclick = () => { page++; load(); };

load();
</script>

"""
    return HTMLResponse(html)

@app.exception_handler(404)
async def not_found(_, __): return PlainTextResponse("No encontrado", status_code=404)
# ==== Admin endpoints (lista de sesiones) ====
import os
import secrets
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import create_engine, text

admin_router = APIRouter(prefix="/admin")
security = HTTPBasic()

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")

def require_basic(credentials: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(credentials.username, ADMIN_USER)
    pass_ok = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (user_ok and pass_ok):
        # obliga al browser/cliente a enviar credenciales Basic
        raise HTTPException(status_code=401, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Basic"})
    return True

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://triage:triage@db:5432/triage")
engine_admin = create_engine(DB_URL, pool_pre_ping=True)

@admin_router.get("/sessions", dependencies=[Depends(require_basic)])
def admin_sessions(
    order_by: str = Query("created_at_desc"),
    limit: int = 50,
    offset: int = 0
):
    order_sql = "created_at DESC" if order_by == "created_at_desc" else "created_at ASC"
    with engine_admin.begin() as con:
        rows = con.execute(text(f"""
            SELECT
              session_id,
              created_at,
              patient_name,
              chief_complaint,
              predicted_esi,
              mapped_manchester
            FROM sessions
            ORDER BY {order_sql}
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset}).mappings().all()
        total = con.execute(text("SELECT COUNT(*) FROM sessions")).scalar_one()
    return {"total": total, "items": [dict(r) for r in rows]}

