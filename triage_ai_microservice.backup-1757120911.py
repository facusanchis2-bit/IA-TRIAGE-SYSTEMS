# triage_ai_microservice.py
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse

APP_VERSION = "0.8.0-identificacion-argente"
DB_PATH = Path(__file__).with_name("triage_sessions.db")
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --------------------- ORDEN DE PREGUNTAS ---------------------
STEP_ORDER = [
    "nombre_apellido",     # 1
    "edad",                # 2
    "domicilio",           # 3
    "ocupacion",           # 4 (¿trabaja? dónde?)
    "antecedentes",        # 5
    "medicacion",          # 6
    "alergias",            # 7
    "factores_riesgo",     # 8
    "sintomas_asoc",       # 9 (cotidiano)
    "alivio",              # 10
    "desencadenantes",     # 11
    "caracter",            # 12
    "intensidad",          # 13 (0–10)
    "localizacion",        # 14
    "dirigidas",           # 15 (dinámicas Argente)
    "consentimiento",      # 16
]

LABELS = {
    "nombre_apellido": "nombre y apellido",
    "edad": "edad",
    "domicilio": "dónde vivís",
    "ocupacion": "trabajo/ocupación",
    "antecedentes": "enfermedades previas",
    "medicacion": "medicación actual",
    "alergias": "alergias",
    "factores_riesgo": "factores de riesgo",
    "sintomas_asoc": "síntomas asociados",
    "alivio": "qué lo alivia",
    "desencadenantes": "qué lo desencadena o empeora",
    "caracter": "cómo lo describís",
    "intensidad": "intensidad (0–10)",
    "localizacion": "dónde lo sentís",
    "dirigidas": "preguntas dirigidas",
    "consentimiento": "consentimiento",
}

# Lenguaje cotidiano
UI_QUESTION = {
    "nombre_apellido": "¿Cuál es tu nombre y apellido?",
    "edad": "¿Qué edad tenés? (en años)",
    "domicilio": "¿En qué ciudad o localidad vivís?",
    "ocupacion": "¿Trabajás? ¿En qué?",
    "antecedentes": "¿Tenés enfermedades previas diagnosticadas? (por ej. presión alta, diabetes). Separá por coma. Si ninguna, escribí 'ninguna'.",
    "medicacion": "¿Qué medicaciones tomás actualmente? Separá por coma. Si ninguna, escribí 'ninguna'.",
    "alergias": "¿Sos alérgico/a a algún medicamento o alimento? Si ninguna, escribí 'no'.",
    "factores_riesgo": "¿Tenés factores de riesgo? (ej. presión alta, diabetes, colesterol, fumás, antecedentes familiares).",
    "sintomas_asoc": "Contanos otros síntomas que notes (por ej. fiebre, tos, te falta el aire, latidos muy fuertes, desmayo, vómitos). Separá por coma.",
    "alivio": "¿Hay algo que lo alivie? (reposo, alguna medicación, cierta postura).",
    "desencadenantes": "¿Qué lo desencadena o empeora? (esfuerzo, comidas, estrés, movimiento).",
    "caracter": "¿Cómo lo describís? (opresivo, punzante, quemante, cólico, etc.)",
    "intensidad": "En una escala de 0 a 10, ¿qué intensidad tiene?",
    "localizacion": "¿Dónde lo sentís principalmente? (pecho, vientre/panza, cabeza, espalda, garganta, orinar, golpe/trauma, etc.)",
    "dirigidas": "Contanos un poco más según el lugar que marcaste.",
    "consentimiento": "Escribí 'sí' para confirmar que leíste la recomendación asistida (no reemplaza la evaluación médica).",
}

MANCHESTER_MAP = {1: "Red", 2: "Orange", 3: "Yellow", 4: "Green", 5: "Blue"}

app = FastAPI(title="Triage Inteligente", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- DB ---------------------
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
        )""")

@app.on_event("startup")
async def _startup():
    init_db()

def save_session_to_db(state: Dict[str, Any]) -> None:
    triage = state.get("final_triage", {})
    gen = state.get("llm_summary", {})
    created_at = state.get("created_at") or datetime.utcnow().isoformat(timespec="seconds")
    state["created_at"] = created_at
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        INSERT OR REPLACE INTO sessions(
            session_id, created_at, patient_json, chief_complaint,
            extracted_entities_json, derived_features_json,
            predicted_esi, mapped_manchester, recommendation,
            override_applied, override_reason, differentials_json,
            shap_top_json, llm_summary_json, disclaimer_ack, state_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            state.get("session_id"),
            created_at,
            json.dumps(state.get("patient", {}), ensure_ascii=False),
            state.get("chief_complaint", ""),
            json.dumps(state.get("extracted_entities", {}), ensure_ascii=False),
            json.dumps(state.get("derived_features", {}), ensure_ascii=False),
            triage.get("predicted_ESI"),
            triage.get("mapped_manchester"),
            triage.get("recommendation"),
            1 if triage.get("override_applied") else 0,
            (triage.get("override_reason") or {}).get("reason") if isinstance(triage.get("override_reason"), dict) else triage.get("override_reason"),
            json.dumps(triage.get("possible_differentials", []), ensure_ascii=False),
            json.dumps(triage.get("shap_top", []), ensure_ascii=False),
            json.dumps(gen, ensure_ascii=False),
            1 if state.get("disclaimer_ack") else 0,
            json.dumps(state, ensure_ascii=False),
        ))

# ------------------ Utils --------------------
def gen_session_id() -> str:
    raw = f"{datetime.utcnow().isoformat()}-{APP_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def append_timeline(st: Dict[str, Any], ev: str, data: Any=None):
    st.setdefault("timeline", []).append({"ts": now_iso(), "event": ev, "data": data})

def normalize(s: str) -> str:
    return (s or "").lower().strip()

# ---------------- NLP/Features ----------------
def simple_ner_and_features(ans: Dict[str, str]) -> Dict[str, Any]:
    txt = " ".join([
        normalize(ans.get("sintomas_asoc","")),
        normalize(ans.get("caracter","")),
        normalize(ans.get("localizacion","")),
        normalize(ans.get("desencadenantes","")),
        normalize(ans.get("alivio","")),
        normalize(ans.get("antecedentes","")),
        normalize(ans.get("factores_riesgo","")),
        normalize(ans.get("dirigidas","")),
    ])

    feat = {
        # cardiopulmonares
        "chest_pain": int(any(k in txt for k in ["pecho","precordial"])),
        "dyspnea": int(any(k in txt for k in ["falta de aire","ahogo"])),
        "palpitations": int(any(k in txt for k in ["latidos fuertes","palpitaciones"])),
        "syncope": int(any(k in txt for k in ["desmayo","se desmayó"])),
        "radiation_left_arm": int(any(k in txt for k in ["brazo izquierdo","mandíbula"])),
        "diaphoresis": int(any(k in txt for k in ["sudor","transpiración fría"])),
        "nausea": int(any(k in txt for k in ["náusea","nausea","vómit","vomit"])),
        "sudden_onset": int(any(k in txt for k in ["inicio súbito","de repente","brusco"])),

        # neurológico y sangrado
        "altered_consciousness": int(any(k in txt for k in ["inconsciente","no responde","confuso","somnoliento"])),
        "focal_deficit_sudden": int(any(k in txt for k in ["hablar","balbucea","cara caída","hemipares","fuerza en brazo"])),
        "massive_bleeding": int(any(k in txt for k in ["sangrado abundante","hemorragia masiva"])),

        # hemodinamia
        "hypotension": int("presión muy baja" in txt or "hipotens" in txt),
        "hemodynamic_instability": int("shock" in txt or "muy pálido" in txt),

        # comorbilidades
        "diabetes": int("diabetes" in txt),
        "htn": int("presión alta" in txt or "hipertens" in txt),
    }

    # Intensidad 0-10 si viene
    try:
        n = float(ans.get("intensidad","").strip())
        feat["severity_num"] = max(0.0, min(10.0, n))
    except Exception:
        feat["severity_num"] = 0.0

    entities = {
        "symptoms": [k for k,v in feat.items() if v and k in ["chest_pain","dyspnea","palpitations","syncope","radiation_left_arm","diaphoresis","nausea","sudden_onset"]],
        "neuro": [k for k,v in feat.items() if v and k in ["altered_consciousness","focal_deficit_sudden"]],
        "bleeding": ["massive_bleeding"] if feat.get("massive_bleeding") else [],
        "comorbidities": [k for k,v in feat.items() if v and k in ["diabetes","htn"]],
    }
    return {"features": feat, "entities": entities}

# -------------- Clasificador ------------------
_WEIGHTS = {
    "chest_pain": 2.5, "dyspnea": 2.0, "radiation_left_arm": 1.5,
    "diaphoresis": 1.0, "nausea": 0.5, "sudden_onset": 1.0,
    "palpitations": 0.5, "syncope": 1.5,
    "diabetes": 0.5, "htn": 0.5,
}
def hard_overrides(feat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if feat.get("massive_bleeding"): return {"esi":1,"reason":"Sangrado abundante"}
    if feat.get("altered_consciousness") and feat.get("focal_deficit_sudden"): return {"esi":1,"reason":"Alteración de conciencia + signos focales"}
    if feat.get("chest_pain") and feat.get("dyspnea") and (feat.get("hypotension") or feat.get("hemodynamic_instability")):
        return {"esi":1,"reason":"Dolor en pecho + falta de aire + inestabilidad"}
    return None

def risk_score(feat: Dict[str, Any]) -> float:
    s = sum(_WEIGHTS[k] for k in _WEIGHTS if feat.get(k))
    s += 0.1 * float(feat.get("severity_num",0))
    return s

def predict_esi(feat: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    ov = hard_overrides(feat)
    if ov:
        return 1, ov, [{"feature": ov["reason"], "value": 1, "contrib": 6.0}]
    s = risk_score(feat)
    if s >= 6.5:   esi = 2
    elif s >= 3.5: esi = 3
    elif s >= 1.5: esi = 4
    else:          esi = 5
    shap = [{"feature": k, "value": 1, "contrib": (_WEIGHTS.get(k,0) or (0.1 if k=="severity_num" else 0))}
            for k in feat if feat.get(k)]
    shap.sort(key=lambda x: -x["contrib"])
    return esi, None, shap[:8]

# --------- Capa generativa placeholder --------
def llm_generate_summary(triage: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    esi = triage.get("predicted_ESI")
    if esi in (1,2):
        studies = ["ECG", "SatO2 continua", "Troponinas", "Monitoreo"]
        risks = ["Posible evento cardiovascular agudo"]
    elif esi == 3:
        studies = ["Laboratorio básico", "ECG si dolor en pecho", "Rx tórax si falta de aire"]
        risks = ["Requiere evaluación en guardia"]
    else:
        studies = ["Control ambulatorio"]
        risks = ["Bajo riesgo aparente"]
    hce = f"Motivo resumido: {state.get('chief_complaint','')}. ESI {esi} ({triage.get('mapped_manchester')})."
    return {
        "suggested_level": f"ESI {esi} ({triage.get('mapped_manchester')})",
        "initial_studies": studies,
        "risks": risks,
        "hce_text": hce,
        "disclaimer": "Recomendación asistida por IA, no reemplaza evaluación médica. No se indican dosis.",
    }

# ------------- Ramas dinámicas (Argente) -------------
def followups_for_location(location: str, answered: bool) -> Optional[str]:
    loc = normalize(location)
    if any(k in loc for k in ["pecho","precordial","torac"]):
        if not answered:
            return "Para el dolor en el pecho: ¿cuándo empezó? ¿Se va al brazo izquierdo o mandíbula? ¿Te falta el aire? ¿Sudor frío? ¿Náuseas?"
    if any(k in loc for k in ["vientre","panza","abdomen","estómago","estomago"]):
        if not answered:
            return "Para dolor de panza: ¿en qué zona? ¿Fiebre? ¿Vómitos? ¿Diarrea o estreñimiento? ¿Embarazo posible?"
    if any(k in loc for k in ["cabeza","cefalea"]):
        if not answered:
            return "Para dolor de cabeza: ¿empezó de golpe? ¿Rigidez de cuello? ¿Dificultad para hablar o mover? ¿Molesta la luz?"
    if any(k in loc for k in ["tos","respirar","garganta","falta de aire"]):
        if not answered:
            return "Síntomas respiratorios: ¿tos con flema? ¿Fiebre? ¿Te falta el aire al caminar o en reposo? ¿Dolor al respirar?"
    if any(k in loc for k in ["orinar","orina","pis","urin"]):
        if not answered:
            return "Al orinar: ¿ardor? ¿Aumentaron las ganas? ¿Sangre en la orina? ¿Dolor en cintura?"
    if any(k in loc for k in ["golpe","trauma","caída","fractura","torcedura","esguince"]):
        if not answered:
            return "Trauma: ¿cómo fue el golpe? ¿Podés apoyar o mover? ¿Deformidad? ¿Dolor con el movimiento? ¿Pérdida de conocimiento?"
    return None

def ui_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    step = state.get("current_step", STEP_ORDER[0])
    idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
    field_type = "text"
    placeholder = "Escribí tu respuesta"
    if step in ("edad","intensidad"):
        field_type = "number"
        placeholder = "Ingresar número"
    title = "Ayudanos a orientarte"
    subtitle = "Te vamos a hacer algunas preguntas rápidas para estimar la urgencia."
    return {
        "current_step": step,
        "current_label": LABELS.get(step, step),
        "step_index": idx+1,
        "step_total": len(STEP_ORDER),
        "field_type": field_type,
        "placeholder": placeholder,
        "title": title,
        "subtitle": subtitle,
    }

def next_question_for(state: Dict[str, Any]) -> str:
    step = state.get("current_step", STEP_ORDER[0])
    if step == "dirigidas":
        loc = state.get("answers", {}).get("localizacion", "")
        q = followups_for_location(loc, answered=False) or UI_QUESTION["dirigidas"]
        return q
    return UI_QUESTION.get(step, "Continuar…")

# ------------------- API ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/triage")
async def triage(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = (body.get("session_id") or "").strip()
    user_input = (body.get("input") or "").strip()
    user_id = body.get("user_id")

    # crear sesión
    if not session_id:
        session_id = gen_session_id()
        state = {
            "session_id": session_id,
            "created_at": now_iso(),
            "answers": {},
            "timeline": [],
            "disclaimer_ack": False,
            "version": APP_VERSION,
            "request_log": [],
            "current_step": STEP_ORDER[0],
        }
        SESSIONS[session_id] = state
        append_timeline(state, "session_started")
        return {"session_id": session_id, "next_question": next_question_for(state), "completed": False, "ui": ui_meta(state)}

    # recuperar estado
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    state["request_log"].append({"ts": now_iso(), "user_id": user_id, "input": user_input, "step": state.get("current_step")})
    step = state.get("current_step", STEP_ORDER[0])
    ans = state.setdefault("answers", {})

    # tolerancia a vacío para no romper envío
    if not user_input:
        defaults = {"alergias":"no","antecedentes":"ninguna","medicacion":"ninguna","ocupacion":"no","factores_riesgo":""}
        user_input = defaults.get(step, "")

    # guardar respuesta
    ans[step] = user_input
    append_timeline(state, "answer", {"step": step, "input": user_input})

    # armar patient
    state["patient"] = {
        "nombre_apellido": ans.get("nombre_apellido"),
        "edad": ans.get("edad"),
        "domicilio": ans.get("domicilio"),
        "ocupacion": ans.get("ocupacion"),
    }

    # chief complaint sintético para el panel
    cc_parts = [ans.get("localizacion",""), ans.get("caracter",""), ans.get("sintomas_asoc","")]
    state["chief_complaint"] = " | ".join([p for p in cc_parts if p]).strip()

    # ¿terminó?
    needs_dirigidas = followups_for_location(ans.get("localizacion",""), answered=bool(ans.get("dirigidas"))) is not None
    base_done = all(ans.get(k) is not None for k in STEP_ORDER if k != "dirigidas")
    if needs_dirigidas and not ans.get("dirigidas"):
        base_done = False

    if base_done and step == "consentimiento":
        # NLP/Features + clasificación
        nlp = simple_ner_and_features(ans)
        features, entities = nlp["features"], nlp["entities"]
        esi, override, shap_top = predict_esi(features)

        if esi == 1:
            diffs = ["Emergencia mayor (sangrado/ACV/SCA complicado)"]
            rec = "Atención inmediata. Monitoreo y equipo de emergencias."
        elif esi == 2:
            diffs = ["Posible síndrome coronario", "Tromboembolismo", "Insuficiencia respiratoria"]
            rec = "Urgencias (ECG, Troponinas, monitoreo)."
        elif esi == 3:
            diffs = ["Dolor en pecho inespecífico", "Reflujo", "Costocondritis"]
            rec = "Evaluación en guardia."
        elif esi == 4:
            diffs = ["Cefalea tensional", "Lumbalgia", "Infección leve"]
            rec = "Atención diferida/ambulatoria."
        else:
            diffs = ["Consulta simple", "Síntomas leves"]
            rec = "Alta con recomendaciones."

        triage_result = {
            "predicted_ESI": esi,
            "mapped_manchester": MANCHESTER_MAP.get(esi),
            "possible_differentials": diffs,
            "recommendation": rec,
            "shap_top": shap_top,
            "override_applied": bool(override),
            "override_reason": override,
        }
        gen = llm_generate_summary(triage_result, state)

        state["extracted_entities"] = entities
        state["derived_features"] = features
        state["final_triage"] = triage_result
        state["llm_summary"] = gen
        state["disclaimer_ack"] = True

        save_session_to_db(state)
        append_timeline(state, "completed", {"esi": esi})

        return {"session_id": state["session_id"], "completed": True, "triage_result": triage_result, "generative_summary": gen}

    # avanzar
    try:
        idx = STEP_ORDER.index(step)
    except ValueError:
        idx = 0

    if step == "localizacion":
        q_dir = followups_for_location(ans.get("localizacion",""), answered=bool(ans.get("dirigidas")))
        if q_dir:
            state["current_step"] = "dirigidas"
            return {"session_id": state["session_id"], "next_question": q_dir, "completed": False, "ui": ui_meta(state)}

    if step == "dirigidas":
        state["current_step"] = "consentimiento"
    else:
        state["current_step"] = STEP_ORDER[min(idx+1, len(STEP_ORDER)-1)]

    return {"session_id": state["session_id"], "next_question": next_question_for(state), "completed": False, "ui": ui_meta(state)}

# ------------- Export & Panel -----------------
@app.get("/admin/sessions")
def list_sessions():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
        SELECT session_id, created_at, chief_complaint,
               predicted_esi, mapped_manchester, recommendation,
               override_applied, override_reason
        FROM sessions
        ORDER BY datetime(created_at) DESC
        """).fetchall()
        return JSONResponse([dict(r) for r in rows])

@app.get("/admin/sessions/{session_id}")
def session_detail(session_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not r: raise HTTPException(404, "not found")
        d = dict(r)
        for k in ["patient_json","extracted_entities_json","derived_features_json","differentials_json","shap_top_json","llm_summary_json","state_json"]:
            if d.get(k):
                try: d[k] = json.loads(d[k])
                except: pass
        return JSONResponse(d)

@app.get("/export/excel")
def export_excel():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
        SELECT
          session_id as Sesion, created_at as Fecha, chief_complaint as Motivo,
          predicted_esi as ESI, mapped_manchester as Manchester, recommendation as Recomendacion,
          override_applied as Override, override_reason as MotivoOverride
        FROM sessions
        ORDER BY datetime(Fecha) DESC
        """, con)
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Resumen")
        buf.seek(0)
        return StreamingResponse(buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="triage_export.xlsx"'}
        )

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    html = """
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
</style>
<div class="top"><h1 style="margin:0;font-size:20px">Panel Médico</h1>
<a class="btn" href="/export/excel">Descargar Excel</a></div>
<table id="t"><thead><tr>
<th>Fecha</th><th>Sesión</th><th>Motivo</th><th>ESI</th><th>Manchester</th><th>Recomendación</th><th>Override</th>
</tr></thead><tbody></tbody></table>
<script>
(async function(){
 const r = await fetch('/admin/sessions'); const rows = await r.json();
 const tb = document.querySelector('#t tbody'); tb.innerHTML='';
 for (const x of rows){
  const tr=document.createElement('tr');
  tr.innerHTML = `
    <td>${x.created_at||''}</td>
    <td><a href="/admin/sessions/${x.session_id}" target="_blank">${x.session_id}</a></td>
    <td>${(x.chief_complaint||'').slice(0,120)}</td>
    <td>${x.predicted_esi ?? ''}</td>
    <td><span class="tag ${x.mapped_manchester||''}">${x.mapped_manchester||''}</span></td>
    <td>${(x.recommendation||'').slice(0,120)}</td>
    <td>${x.override_applied ? 'Sí' : 'No'}</td>`;
  tb.appendChild(tr);
 }
})();
</script>
"""
    return HTMLResponse(html)

@app.exception_handler(404)
async def not_found(_, __): return PlainTextResponse("No encontrado", status_code=404)
