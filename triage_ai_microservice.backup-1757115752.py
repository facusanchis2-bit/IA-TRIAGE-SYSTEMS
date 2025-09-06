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
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
APP_VERSION = "0.5.0-medpanel"
DB_PATH = Path(__file__).with_name("triage_sessions.db")

# almacenamiento en memoria del estado de entrevistas en curso
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Pasos (orden mínimo — el árbol adaptativo puede refinar la “next_question”)
STEP_ORDER = [
    "identificacion",
    "chief_complaint",
    "medicacion",
    "antecedentes",
    "factores_riesgo",
    "consentimiento",
]

MANCHESTER_MAP = {1: "Red", 2: "Orange", 3: "Yellow", 4: "Green", 5: "Blue"}

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Triage Inteligente", version=APP_VERSION)

# CORS (en dev: *; en prod, restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ limitar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# DB: inicialización y helpers
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

    created_at = state.get("created_at")
    if not created_at:
        created_at = datetime.utcnow().isoformat(timespec="seconds")
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
# Utilidades: ids, logs simples
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
# NLP/Features (versión simple, ajustable)
# -----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    return (s or "").lower().strip()


def simple_ner_and_features(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Deriva features binarios a partir de texto libre muy simple.
    Ajustar con spaCy/scispaCy/transformer cuando tengas el modelo.
    """
    cc = normalize_text(answers.get("chief_complaint", ""))
    follow = normalize_text(answers.get("adaptive_followups", ""))
    meds = normalize_text(answers.get("medicacion", ""))
    antecedentes = normalize_text(answers.get("antecedentes", ""))
    riesgo = normalize_text(answers.get("factores_riesgo", ""))

    txt = " ".join([cc, follow, meds, antecedentes, riesgo])

    feat = {
        # cardiaco
        "chest_pain": int(any(k in txt for k in ["pecho", "torác", "torac"])),
        "dyspnea": int(any(k in txt for k in ["disnea", "falta de aire", "ahogo"])),
        "radiation_left_arm": int(any(k in txt for k in ["brazo izq", "brazo izquierdo", "irradiado"])),
        "diaphoresis": int(any(k in txt for k in ["sudor", "diafores"])),
        "nausea": int("náuse" in txt or "nausea" in txt),
        "sudden_onset": int(any(k in txt for k in ["inicio súbito", "de repente", "brusco"])),

        # neurológico / hemorragia
        "altered_consciousness": int(any(k in txt for k in ["inconsciente", "no responde", "somnoliento"])),
        "focal_deficit_sudden": int(any(k in txt for k in ["déficit focal", "dificultad al hablar", "hemiparesia", "cara caída"])),
        "massive_bleeding": int(any(k in txt for k in ["hemorragia masiva", "sangrado abundante", "sangra mucho"])),

        # hipotensión/ inestabilidad (si luego agregás SV reales, cambiá esto)
        "hypotension": int("hipotension" in txt or "hipotensión" in txt),
        "hemodynamic_instability": int(any(k in txt for k in ["inestabilidad hemodin", "shock"])),

        # comorbilidades
        "diabetes": int("dm2" in antecedentes or "diabetes" in antecedentes),
        "htn": int("hta" in antecedentes or "hipertens" in antecedentes),

        # edad (si se capturó en texto de identificación)
        "age_over_65": int(any(k in answers.get("identificacion", "") for k in [" 65", " 66", " 70", " 80"]))  # placeholder
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
# Clasificador: hard-rules + score escalonado (corrige “todo ESI1”)
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
    """ESI 1 solo si hay emergencias obvias."""
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
    s = 0.0
    for k, w in _WEIGHTS.items():
        s += w * (1 if feat.get(k) == 1 else 0)
    return s


def predict_esi(feat: Dict[str, int]) -> (int, Optional[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Devuelve (esi, override, shap_top). ESI 1 SOLO por hard-rules.
    Score (s) -> ESI:
      s>=6.5 -> 2
      3.5–6.49 -> 3
      1.5–3.49 -> 4
      <1.5 -> 5
    """
    ov = apply_hard_overrides(feat)
    if ov:
        return 1, ov, [{"feature": ov["reason"], "value": 1, "contrib": 5.0}]

    s = risk_score(feat)
    if s >= 6.5:
        esi = 2
    elif s >= 3.5:
        esi = 3
    elif s >= 1.5:
        esi = 4
    else:
        esi = 5

    shap = []
    for k, w in _WEIGHTS.items():
        if feat.get(k) == 1:
            shap.append({"feature": k, "value": 1, "contrib": w})
    shap.sort(key=lambda x: -abs(x["contrib"]))
    return esi, None, shap[:8]


# -----------------------------------------------------------------------------
# Generación de sugerencias (LLM placeholder con guardrails)
# -----------------------------------------------------------------------------
def llm_generate_summary(triage: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder seguro: NO sugiere dosis ni diagnósticos definitivos.
    """
    esi = triage.get("predicted_ESI")
    studies = []
    risks = []
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
        f"Resumen: Motivo principal '{state.get('chief_complaint','')}'. "
        f"Clasificación ESI {esi} ({triage.get('mapped_manchester')}). "
        "Esta recomendación es asistida y no reemplaza el juicio clínico."
    )
    return {
        "suggested_level": f"ESI {esi} ({triage.get('mapped_manchester')})",
        "initial_studies": studies,
        "risks": risks,
        "hce_text": hce_text,
        "disclaimer": "Esta recomendación es asistida por IA y no reemplaza la evaluación profesional. No se indican dosis farmacológicas.",
    }


# -----------------------------------------------------------------------------
# Árbol adaptativo mínimo (pregunta siguiente)
# -----------------------------------------------------------------------------
def followup_question_if_needed(state: Dict[str, Any]) -> Optional[str]:
    answers = state.get("answers", {})
    cc = (answers.get("chief_complaint") or "").lower()
    if any(k in cc for k in ["pecho","torác","torac"]):
        if not answers.get("adaptive_followups"):
            return "Describa el dolor torácico (inicio, irradiación, disnea, sudoración, náuseas)."
    if any(k in cc for k in ["abdomen","estómago","estomago"]):
        if not answers.get("adaptive_followups"):
            return "Localización del dolor abdominal, fiebre, vómitos, cambios en deposiciones."
    if any(k in cc for k in ["cabeza","cefalea"]):
        if not answers.get("adaptive_followups"):
            return "Inicio súbito, rigidez de cuello, déficit neurológico, fotofobia."
    return None

def next_question_for(state: Dict[str, Any]) -> str:
    labels = {
        "identificacion": "identificación",
        "chief_complaint": "motivo de consulta",
        "medicacion": "medicación actual",
        "antecedentes": "antecedentes médicos",
        "factores_riesgo": "factores de riesgo",
        "consentimiento": "consentimiento",
    }
    step = state.get("current_step", STEP_ORDER[0])
    idx = max(0, STEP_ORDER.index(step))
    if idx < len(STEP_ORDER) - 1:
        nxt = STEP_ORDER[idx + 1]
        return f"Ingrese {labels.get(nxt, nxt.replace('_',' '))}"
    return "¿Desea confirmar y finalizar?"
# 5) si no terminó: calcular siguiente pregunta
# mover puntero de step
try:
    idx = STEP_ORDER.index(step)
    state["current_step"] = STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]
except ValueError:
    state["current_step"] = STEP_ORDER[0]

# Si el motivo de consulta requiere followups y no se respondieron, forzamos ese paso
fq = followup_question_if_needed(state)
if fq:
    state["current_step"] = "adaptive_followups"
    q = fq
else:
    q = next_question_for(state)

return {
    "session_id": session_id,
    "next_question": q,
    "completed": False,
    "state_snapshot": {"session_id": session_id}
}

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/triage")
async def triage(request: Request):
    """
    Flujo:
    - Si no hay session_id: crea sesión y devuelve primera pregunta.
    - Si hay session_id + step + input: guarda, avanza, y al final:
      calcula features -> clasifica -> guarda en SQLite -> responde resultado final.
    """
    body = await request.json() if request.body else {}
    session_id = body.get("session_id")
    step = body.get("step")
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
        }
        SESSIONS[session_id] = state
        append_timeline(state, "session_started")
        state["current_step"] = STEP_ORDER[0]
        q = "Ingrese identificación (Nombre Apellido, edad, ciudad, ocupación)."
        return {"session_id": session_id, "next_question": q, "completed": False, "state_snapshot": {"session_id": session_id}}

    # 2) recuperar estado
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    # log simple
    state["request_log"].append({"ts": now_iso(), "user_id": user_id, "step": step, "input": user_input})

    # 3) guardar respuesta del paso
    if step not in STEP_ORDER:
        # permitir clientes muy simples: si no envían step, usamos current_step
        step = state.get("current_step", STEP_ORDER[0])

    answers = state.setdefault("answers", {})

    if step == "identificacion":
        answers["identificacion"] = user_input
        # parsing mínimo: solo chief later
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

    # 4) decidir si ya terminamos
    # criterio simple: cuando completamos todos los pasos, finalizamos
    all_done = all(k in answers and answers[k] for k in STEP_ORDER if k != "consentimiento") and bool(answers.get("consentimiento"))
    if all_done:
        # NLP/Features
        nlp = simple_ner_and_features(answers)
        features = nlp["features"]
        entities = nlp["entities"]

        # clasificación
        esi, override, shap_top = predict_esi(features)

        # diferenciales/recomendación (muy básicos; ajustá según tus ramas)
        diffs = []
        rec = "Evaluación en guardia."
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

        # generativo (placeholder seguro)
        gen = llm_generate_summary(triage_result, state)

        # guardar en estado y persistir
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

    # 5) si no terminó: calcular siguiente pregunta
    # mover puntero de step
    try:
        idx = STEP_ORDER.index(step)
        state["current_step"] = STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]
    except ValueError:
        state["current_step"] = STEP_ORDER[0]

    q = next_question_for(state)
    return {"session_id": session_id, "next_question": q, "completed": False, "state_snapshot": {"session_id": session_id}}


# -----------------------------------------------------------------------------
# Export por sesión (JSON / CSV)
# -----------------------------------------------------------------------------
@app.get("/export/{session_id}")
def export_session(session_id: str, format: str = Query("json", regex="^(json|csv)$")):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")

        d = dict(row)
        if format == "json":
            # devolver JSON “limpio”
            for k in [
                "patient_json",
                "extracted_entities_json",
                "derived_features_json",
                "differentials_json",
                "shap_top_json",
                "llm_summary_json",
                "state_json",
            ]:
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        pass
            return JSONResponse(d)

        # CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["campo", "valor"])
        for k, v in d.items():
            writer.writerow([k, v])
        output.seek(0)
        return StreamingResponse(
            iter([output.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{session_id}.csv"'},
        )


# -----------------------------------------------------------------------------
# Panel médico y Excel
# -----------------------------------------------------------------------------
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
        for k in [
            "patient_json",
            "extracted_entities_json",
            "derived_features_json",
            "differentials_json",
            "shap_top_json",
            "llm_summary_json",
            "state_json",
        ]:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
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
        return StreamingResponse(
            buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers
        )


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    html = """
<!doctype html>
<html lang="es">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panel Médico</title>
<style>
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#0f172a; }
  table { border-collapse: collapse; width:100%; }
  th, td { border:1px solid #e2e8f0; padding:8px 10px; font-size:14px; }
  th { background:#f8fafc; text-align:left; }
  .top { display:flex; gap:12px; align-items:center; margin-bottom:12px; }
  .btn { display:inline-block; padding:8px 12px; background:#0f172a; color:white; text-decoration:none; border-radius:8px; }
  .tag { padding:2px 8px; border-radius:9999px; font-size:12px; }
  .red{ background:#fee2e2; color:#991b1b }
  .orange{ background:#ffedd5; color:#9a3412 }
  .yellow{ background:#fef9c3; color:#854d0e }
  .green{ background:#dcfce7; color:#166534 }
  .blue{ background:#dbeafe; color:#1e40af }
</style>
<div class="top">
  <h1 style="margin:0;font-size:18px;">Panel Médico</h1>
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


# -----------------------------------------------------------------------------
# Fallback 404 plano (más legible en dev)
# -----------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return PlainTextResponse("No encontrado", status_code=404)
