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

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
APP_VERSION = "0.6.0-medpanel-ui"
DB_PATH = Path(__file__).with_name("triage_sessions.db")

# Estado en memoria para sesiones activas
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Flujo base (sin followups; estos son condicionales)
STEP_ORDER = [
    "identificacion",
    "chief_complaint",
    "medicacion",
    "antecedentes",
    "factores_riesgo",
    "consentimiento",
]

# Etiquetas legibles por UI
LABELS = {
    "identificacion": "identificación",
    "chief_complaint": "motivo de consulta",
    "adaptive_followups": "preguntas dirigidas",
    "medicacion": "medicación actual",
    "antecedentes": "antecedentes médicos",
    "factores_riesgo": "factores de riesgo",
    "consentimiento": "consentimiento",
}

MANCHESTER_MAP = {1: "Red", 2: "Orange", 3: "Yellow", 4: "Green", 5: "Blue"}

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Triage Inteligente", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # ⚠️ limitar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
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
            """
        )

@app.on_event("startup")
async def on_startup():
    init_db()

def save_session_to_db(state: Dict[str, Any]) -> None:
    triage = state.get("final_triage", {})
    gen = state.get("llm_summary", {})

    created_at = state.get("created_at") or datetime.utcnow().isoformat(timespec="seconds")
    state["created_at"] = created_at

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id, created_at, patient_json, chief_complaint,
                extracted_entities_json, derived_features_json,
                predicted_esi, mapped_manchester, recommendation,
                override_applied, override_reason, differentials_json,
                shap_top_json, llm_summary_json, disclaimer_ack, state_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
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
                (triage.get("override_reason") or {}).get("reason")
                if isinstance(triage.get("override_reason"), dict)
                else triage.get("override_reason"),
                json.dumps(triage.get("possible_differentials", []), ensure_ascii=False),
                json.dumps(triage.get("shap_top", []), ensure_ascii=False),
                json.dumps(gen, ensure_ascii=False),
                1 if state.get("disclaimer_ack") else 0,
                json.dumps(state, ensure_ascii=False),
            ),
        )

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def gen_session_id() -> str:
    raw = f"{datetime.utcnow().isoformat()}-{APP_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def append_timeline(state: Dict[str, Any], event: str, data: Any = None) -> None:
    state.setdefault("timeline", [])
    state["timeline"].append({"ts": now_iso(), "event": event, "data": data})

# -----------------------------------------------------------------------------
# NLP / Features (placeholder robusto)
# -----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    return (s or "").lower().strip()

def simple_ner_and_features(answers: Dict[str, str]) -> Dict[str, Any]:
    cc = normalize_text(answers.get("chief_complaint", ""))
    follow = normalize_text(answers.get("adaptive_followups", ""))
    antecedentes = normalize_text(answers.get("antecedentes", ""))
    riesgo = normalize_text(answers.get("factores_riesgo", ""))

    txt = " ".join([cc, follow, antecedentes, riesgo])

    feat = {
        "chest_pain": int(any(k in txt for k in ["pecho", "torác", "torac"])),
        "dyspnea": int(any(k in txt for k in ["disnea", "falta de aire", "ahogo"])),
        "radiation_left_arm": int(any(k in txt for k in ["brazo izq", "brazo izquierdo", "irradiado"])),
        "diaphoresis": int(any(k in txt for k in ["sudor", "diafores"])),
        "nausea": int("náuse" in txt or "nausea" in txt),
        "sudden_onset": int(any(k in txt for k in ["inicio súbito", "de repente", "brusco"])),

        "altered_consciousness": int(any(k in txt for k in ["inconsciente", "no responde", "somnoliento"])),
        "focal_deficit_sudden": int(any(k in txt for k in ["déficit focal", "dificultad al hablar", "hemiparesia", "cara caída"])),
        "massive_bleeding": int(any(k in txt for k in ["hemorragia masiva", "sangrado abundante", "sangra mucho"])),

        "hypotension": int("hipotension" in txt or "hipotensión" in txt),
        "hemodynamic_instability": int(any(k in txt for k in ["inestabilidad hemodin", "shock"])),

        "diabetes": int("dm2" in antecedentes or "diabetes" in antecedentes),
        "htn": int("hta" in antecedentes or "hipertens" in antecedentes),

        "age_over_65": 0,  # placeholder si querés parsear edad de identificación
    }

    entities = {
        "symptoms": [k for k, v in feat.items() if v == 1 and k in
                     ["chest_pain", "dyspnea", "radiation_left_arm", "diaphoresis", "nausea", "sudden_onset"]],
        "neuro_flags": [k for k, v in feat.items() if v == 1 and k in
                        ["altered_consciousness", "focal_deficit_sudden"]],
        "bleeding": ["massive_bleeding"] if feat.get("massive_bleeding") else [],
        "comorbidities": [k for k, v in feat.items() if v == 1 and k in ["diabetes", "htn"]],
    }

    return {"features": feat, "entities": entities}

# -----------------------------------------------------------------------------
# Clasificador (hard-rules + score escalonado)
# -----------------------------------------------------------------------------
_WEIGHTS = {
    "age_over_65": 2.0,
    "chest_pain": 2.5,
    "dyspnea": 2.0,
    "radiation_left_arm": 1.5,
    "diaphoresis": 1.0,
    "nausea": 0.5,
    "sudden_onset": 1.0,
    "diabetes": 0.5,
    "htn": 0.5,
}

def apply_hard_overrides(feat: Dict[str, int]) -> Optional[Dict[str, Any]]:
    if feat.get("massive_bleeding") == 1:
        return {"esi": 1, "reason": "Hemorragia masiva"}
    if feat.get("altered_consciousness") == 1 and feat.get("focal_deficit_sudden") == 1:
        return {"esi": 1, "reason": "Alteración de conciencia + déficit focal súbito"}
    if feat.get("chest_pain") == 1 and feat.get("dyspnea") == 1 and (
        feat.get("hypotension") == 1 or feat.get("hemodynamic_instability") == 1
    ):
        return {"esi": 1, "reason": "Dolor torácico + disnea + inestabilidad hemodinámica"}
    return None

def risk_score(feat: Dict[str, int]) -> float:
    return sum(_WEIGHTS[k] for k, w in _WEIGHTS.items() if feat.get(k) == 1)

def predict_esi(feat: Dict[str, int]) -> Tuple[int, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    ov = apply_hard_overrides(feat)
    if ov:
        return 1, ov, [{"feature": ov["reason"], "value": 1, "contrib": 5.0}]

    s = 0.0
    for k, w in _WEIGHTS.items():
        s += w * (1 if feat.get(k) == 1 else 0)

    if s >= 6.5:
        esi = 2
    elif s >= 3.5:
        esi = 3
    elif s >= 1.5:
        esi = 4
    else:
        esi = 5

    shap = [{"feature": k, "value": 1, "contrib": w} for k, w in _WEIGHTS.items() if feat.get(k) == 1]
    shap.sort(key=lambda x: -abs(x["contrib"]))
    return esi, None, shap[:8]

# -----------------------------------------------------------------------------
# Capa generativa placeholder (sin dosis)
# -----------------------------------------------------------------------------
def llm_generate_summary(triage: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    esi = triage.get("predicted_ESI")
    studies, risks = [], []
    if esi in (1, 2):
        studies = ["ECG", "SatO2 continua", "Troponinas", "Monitoreo"]
        risks = ["Posible evento cardiovascular agudo"]
    elif esi == 3:
        studies = ["Laboratorio básico", "ECG si dolor torácico", "Rx tórax si disnea"]
        risks = ["Patología que requiere evaluación en guardia"]
    else:
        studies = ["Control ambulatorio", "Analgesia simple si corresponde"]
        risks = ["Bajo riesgo aparente"]

    hce_text = (
        f"Motivo: '{state.get('chief_complaint','')}'. "
        f"Clasificación ESI {esi} ({triage.get('mapped_manchester')}). "
        "Recomendación asistida; no reemplaza juicio clínico."
    )
    return {
        "suggested_level": f"ESI {esi} ({triage.get('mapped_manchester')})",
        "initial_studies": studies,
        "risks": risks,
        "hce_text": hce_text,
        "disclaimer": "Esta recomendación es asistida por IA y no reemplaza la evaluación profesional. No se indican dosis farmacológicas.",
    }

# -----------------------------------------------------------------------------
# Follow-ups condicionales + próxima pregunta
# -----------------------------------------------------------------------------
def followup_question_if_needed(state: Dict[str, Any]) -> Optional[str]:
    answers = state.get("answers", {})
    cc = (answers.get("chief_complaint") or "").lower()
    if any(k in cc for k in ["pecho", "torác", "torac"]):
        if not answers.get("adaptive_followups"):
            return "Describa el dolor torácico (inicio, irradiación, disnea, sudoración, náuseas)."
    if any(k in cc for k in ["abdomen", "estómago", "estomago"]):
        if not answers.get("adaptive_followups"):
            return "Localización del dolor abdominal, fiebre, vómitos, cambios en deposiciones."
    if any(k in cc for k in ["cabeza", "cefalea"]):
        if not answers.get("adaptive_followups"):
            return "Inicio súbito, rigidez de cuello, déficit neurológico, fotofobia."
    return None

def next_question_for(state: Dict[str, Any]) -> str:
    step = state.get("current_step", STEP_ORDER[0])
    idx = max(0, STEP_ORDER.index(step))
    if idx < len(STEP_ORDER) - 1:
        nxt = STEP_ORDER[idx + 1]
        return f"Ingrese {LABELS.get(nxt, nxt.replace('_',' '))}"
    return "¿Desea confirmar y finalizar?"

def ui_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    step = state.get("current_step", STEP_ORDER[0])
    idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
    # total base + 1 si falta un followup
    needs_f = followup_question_if_needed(state) is not None
    total = len(STEP_ORDER) + (1 if (needs_f and step != "adaptive_followups" and not state.get("answers", {}).get("adaptive_followups")) else 0)
    return {
        "current_step": step,
        "current_label": LABELS.get(step, step),
        "step_index": idx,
        "step_total": total,
    }

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
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

    # 1) crear sesión
    if not session_id:
        session_id = gen_session_id()
        state = {
            "session_id": session_id,
            "created_at": now_iso(),
            "patient": {},
            "answers": {},
            "timeline": [],
            "disclaimer_ack": False,
            "version": APP_VERSION,
            "request_log": [],
            "current_step": STEP_ORDER[0],
        }
        SESSIONS[session_id] = state
        append_timeline(state, "session_started")
        q = "Ingrese identificación (Nombre Apellido, edad, ciudad, ocupación)."
        return {
            "session_id": session_id,
            "next_question": q,
            "completed": False,
            "ui": ui_meta(state),
        }

    # 2) recuperar estado
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    state["request_log"].append({"ts": now_iso(), "user_id": user_id, "input": user_input, "step": state.get("current_step")})

    # 3) guardar respuesta del paso actual
    step = state.get("current_step", STEP_ORDER[0])
    answers = state.setdefault("answers", {})

    if step == "identificacion":
        answers["identificacion"] = user_input
        state["patient"] = {"raw": user_input}
    elif step == "chief_complaint":
        answers["chief_complaint"] = user_input
        state["chief_complaint"] = user_input
    elif step == "adaptive_followups":
        answers["adaptive_followups"] = user_input
    elif step == "medicacion":
        answers["medicacion"] = user_input
    elif step == "antecedentes":
        answers["antecedentes"] = user_input
    elif step == "factores_riesgo":
        answers["factores_riesgo"] = user_input
    elif step == "consentimiento":
        answers["consentimiento"] = user_input
        state["disclaimer_ack"] = True

    append_timeline(state, "answer", {"step": step, "input": user_input})

    # 4) ¿terminó?
    base_done = all(answers.get(k) for k in STEP_ORDER if k != "consentimiento") and bool(answers.get("consentimiento"))
    requires_fup = followup_question_if_needed({"answers": {"chief_complaint": answers.get("chief_complaint", ""), "adaptive_followups": answers.get("adaptive_followups")}}) is not None
    if requires_fup and not answers.get("adaptive_followups"):
        base_done = False

    if base_done:
        nlp = simple_ner_and_features(answers)
        features = nlp["features"]
        entities = nlp["entities"]

        esi, override, shap_top = predict_esi(features)

        if esi == 1:
            diffs = ["Paro/Choque/Hemorragia masiva", "ACV", "SCA complicado"]
            rec = "Atención inmediata. Monitoreo y equipo de emergencias."
        elif esi == 2:
            diffs = ["Síndrome coronario agudo", "Tromboembolismo", "Insuficiencia respiratoria"]
            rec = "Derivar a urgencias (ECG, Troponinas, monitoreo)."
        elif esi == 3:
            diffs = ["Dolor torácico no específico", "Reflujo", "Costocondritis"]
            rec = "Evaluación en guardia. ECG si dolor torácico, Rx si disnea."
        elif esi == 4:
            diffs = ["Cefalea tensional", "Lumbalgia", "Infección leve"]
            rec = "Atención diferida/ambulatoria."
        else:
            diffs = ["Consulta administrativa", "Síntomas leves autolimitados"]
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

        save_session_to_db(state)
        append_timeline(state, "completed", {"esi": esi})

        return {
            "session_id": state["session_id"],
            "completed": True,
            "triage_result": triage_result,
            "generative_summary": gen,
        }

    # 5) si no terminó: avanzar
    if step == "chief_complaint":
        fq = followup_question_if_needed(state)
        if fq and not answers.get("adaptive_followups"):
            state["current_step"] = "adaptive_followups"
            return {"session_id": state["session_id"], "next_question": fq, "completed": False, "ui": ui_meta(state)}

    if step == "adaptive_followups":
        state["current_step"] = "medicacion"
    else:
        try:
            idx = STEP_ORDER.index(step)
            state["current_step"] = STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]
        except ValueError:
            state["current_step"] = STEP_ORDER[0]

    q = next_question_for(state)
    return {"session_id": state["session_id"], "next_question": q, "completed": False, "ui": ui_meta(state)}

# -----------------------------------------------------------------------------
# Export & Panel
# -----------------------------------------------------------------------------
@app.get("/export/{session_id}")
def export_session(session_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        d = dict(row)
        if format == "json":
            for k in ["patient_json","extracted_entities_json","derived_features_json","differentials_json","shap_top_json","llm_summary_json","state_json"]:
                if d.get(k):
                    try: d[k] = json.loads(d[k])
                    except: pass
            return JSONResponse(d)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["campo", "valor"])
        for k, v in d.items():
            writer.writerow([k, v])
        output.seek(0)
        return StreamingResponse(iter([output.read()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{session_id}.csv"'})

@app.get("/admin/sessions")
def list_sessions():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT session_id, created_at, chief_complaint,
                   predicted_esi, mapped_manchester, recommendation,
                   override_applied, override_reason
            FROM sessions
            ORDER BY datetime(created_at) DESC
            """
        ).fetchall()
        return JSONResponse([dict(r) for r in rows])

@app.get("/admin/sessions/{session_id}")
def get_session_detail(session_id: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not r:
            return JSONResponse({"error": "not found"}, status_code=404)
        d = dict(r)
        for k in ["patient_json","extracted_entities_json","derived_features_json","differentials_json","shap_top_json","llm_summary_json","state_json"]:
            if d.get(k):
                try: d[k] = json.loads(d[k])
                except: pass
        return JSONResponse(d)

@app.get("/export/excel")
def export_excel():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT
              session_id as Sesion,
              created_at as Fecha,
              chief_complaint as Motivo,
              predicted_esi as ESI,
              mapped_manchester as Manchester,
              recommendation as Recomendacion,
              override_applied as Override,
              override_reason as MotivoOverride
            FROM sessions
            ORDER BY datetime(Fecha) DESC
            """,
            con,
        )
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Resumen")
        buf.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="triage_export.xlsx"'}
        return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    html = """
<!doctype html>
<html lang="es">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panel Médico</title>
<style>
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#0f172a; background:#f8fafc; }
  table { border-collapse: collapse; width:100%; background:white; border-radius:12px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.05); }
  th, td { border-bottom:1px solid #e2e8f0; padding:10px 12px; font-size:14px; }
  th { background:#f1f5f9; text-align:left; }
  .top { display:flex; gap:12px; align-items:center; margin-bottom:16px; }
  .btn { display:inline-block; padding:10px 14px; background:#0f172a; color:white; text-decoration:none; border-radius:10px; }
  .tag { padding:2px 8px; border-radius:9999px; font-size:12px; }
  .red{ background:#fee2e2; color:#991b1b }
  .orange{ background:#ffedd5; color:#9a3412 }
  .yellow{ background:#fef9c3; color:#854d0e }
  .green{ background:#dcfce7; color:#166534 }
  .blue{ background:#dbeafe; color:#1e40af }
</style>
<div class="top">
  <h1 style="margin:0;font-size:20px;">Panel Médico</h1>
  <a class="btn" href="/export/excel">Descargar Excel</a>
</div>
<table id="t"><thead>
  <tr><th>Fecha</th><th>Sesión</th><th>Motivo</th><th>ESI</th><th>Manchester</th><th>Recomendación</th><th>Override</th></tr>
</thead><tbody></tbody></table>
<script>
async function go(){
  const res = await fetch('/admin/sessions');
  const rows = await res.json();
  const tb = document.querySelector('#t tbody');
  tb.innerHTML = '';
  for (const r of rows){
    const tr = document.createElement('tr');
    const color = (r.mapped_manchester||'').toLowerCase();
    tr.innerHTML = `
      <td>${r.created_at||''}</td>
      <td><a href="/admin/sessions/${r.session_id}" target="_blank">${r.session_id}</a></td>
      <td>${(r.chief_complaint||'').slice(0,120)}</td>
      <td>${r.predicted_esi ?? ''}</td>
      <td><span class="tag ${color}">${r.mapped_manchester||''}</span></td>
      <td>${(r.recommendation||'').slice(0,120)}</td>
      <td>${r.override_applied ? 'Sí' : 'No'}</td>`;
    tb.appendChild(tr);
  }
}
go();
</script>
"""
    return HTMLResponse(content=html)

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return PlainTextResponse("No encontrado", status_code=404)
