# triage_ai_microservice.py — v0.9.0 plan dinámico por motivo
from __future__ import annotations

import hashlib, json, sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

APP_VERSION = "0.9.0-motivo-dinamico"
DB_PATH = Path(__file__).with_name("triage_sessions.db")
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------- PREGUNTAS (texto para UI, lenguaje cotidiano) ----------
UI_QUESTION = {
    "nombre_apellido": "¿Cuál es tu nombre y apellido?",
    "edad": "¿Qué edad tenés? (en años)",
    "domicilio": "¿En qué ciudad o localidad vivís?",
    "ocupacion": "¿Trabajás? ¿En qué?",
    "motivo": "En tus palabras, ¿cuál es el motivo de tu consulta hoy?",
    "antecedentes": "¿Tenés enfermedades previas diagnosticadas? (por ej. presión alta, diabetes). Separá por coma. Si ninguna, escribí 'ninguna'.",
    "medicacion": "¿Qué medicaciones tomás actualmente? Separá por coma. Si ninguna, escribí 'ninguna'.",
    "alergias": "¿Sos alérgico/a a algún medicamento o alimento? Si ninguna, escribí 'no'.",
    "factores_riesgo": "¿Tenés factores de riesgo? (presión alta, diabetes, colesterol, fumás, antecedentes familiares).",
    # bloque común
    "sintomas_asoc": "Contanos otros síntomas que notes (por ej. fiebre, tos, te falta el aire, latidos muy fuertes, desmayo, vómitos). Separá por coma.",
    "alivio": "¿Hay algo que lo alivie? (reposo, alguna medicación, cierta postura).",
    "desencadenantes": "¿Qué lo desencadena o empeora? (esfuerzo, comidas, estrés, movimiento).",
    "caracter": "¿Cómo lo describís? (opresivo, punzante, quemante, cólico, etc.)",
    "intensidad": "En una escala de 0 a 10, ¿qué intensidad tiene?",
    "localizacion": "¿Dónde lo sentís principalmente? (pecho, panza, cabeza, espalda, garganta, al orinar, golpe/trauma, etc.)",
    # dirigidas por rama
    "dirigidas_pecho": "Para dolor en el pecho: ¿cuándo empezó? ¿Se va al brazo izquierdo o mandíbula? ¿Te falta el aire? ¿Sudor frío? ¿Náuseas?",
    "dirigidas_panza": "Para dolor de panza: ¿en qué zona? ¿Fiebre? ¿Vómitos? ¿Diarrea o estreñimiento? ¿Podrías estar embarazada?",
    "dirigidas_cabeza": "Para dolor de cabeza: ¿empezó de golpe? ¿Cuello duro? ¿Dificultad para hablar o mover un lado? ¿Molesta la luz?",
    "dirigidas_resp": "Síntomas respiratorios: ¿tos con flema? ¿Fiebre? ¿Te falta el aire al caminar o en reposo? ¿Dolor al respirar?",
    "dirigidas_urina": "Al orinar: ¿ardor? ¿Vas más veces de lo normal? ¿Sangre en la orina? ¿Dolor en la cintura?",
    "dirigidas_trauma": "Trauma/golpe: ¿cómo fue? ¿Podés apoyar o mover? ¿Deformidad visible? ¿Perdiste el conocimiento?",
    "dirigidas_general": "Contanos lo que creas importante sobre tu síntoma principal (inicio, duración, si cambia con algo).",
    "consentimiento": "Escribí 'sí' para confirmar que leíste la recomendación asistida (no reemplaza la evaluación médica).",
}

LABELS = {
    "nombre_apellido":"nombre y apellido","edad":"edad","domicilio":"dónde vivís","ocupacion":"trabajo/ocupación","motivo":"motivo de consulta",
    "antecedentes":"enfermedades previas","medicacion":"medicación actual","alergias":"alergias","factores_riesgo":"factores de riesgo",
    "sintomas_asoc":"síntomas asociados","alivio":"qué lo alivia","desencadenantes":"qué lo empeora","caracter":"cómo lo describís",
    "intensidad":"intensidad (0–10)","localizacion":"dónde lo sentís","dirigidas":"preguntas dirigidas","consentimiento":"consentimiento"
}

MANCHESTER_MAP = {1:"Red",2:"Orange",3:"Yellow",4:"Green",5:"Blue"}

# ---------- App ----------
app = FastAPI(title="Triage Inteligente", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------- DB ----------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
          session_id TEXT PRIMARY KEY, created_at TEXT,
          patient_json TEXT, chief_complaint TEXT,
          extracted_entities_json TEXT, derived_features_json TEXT,
          predicted_esi INTEGER, mapped_manchester TEXT,
          recommendation TEXT, override_applied INTEGER, override_reason TEXT,
          differentials_json TEXT, shap_top_json TEXT,
          llm_summary_json TEXT, disclaimer_ack INTEGER, state_json TEXT
        );""")

@app.on_event("startup")
async def _on_start():
    init_db()

# ---------- Utils ----------
def gen_session_id()->str:
    raw=f"{datetime.utcnow().isoformat()}-{APP_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def now_iso()->str: return datetime.utcnow().isoformat(timespec="seconds")

def normalize(x:str)->str: return (x or "").lower().strip()

def append_timeline(st:Dict[str,Any], ev:str, data:Any=None):
    st.setdefault("timeline", []).append({"ts": now_iso(), "event": ev, "data": data})

def save_session_to_db(state:Dict[str,Any]):
    triage = state.get("final_triage", {})
    gen = state.get("llm_summary", {})
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
            state["session_id"], state.get("created_at") or now_iso(),
            json.dumps(state.get("patient",{}), ensure_ascii=False),
            state.get("chief_complaint",""),
            json.dumps(state.get("extracted_entities",{}), ensure_ascii=False),
            json.dumps(state.get("derived_features",{}), ensure_ascii=False),
            triage.get("predicted_ESI"), triage.get("mapped_manchester"),
            triage.get("recommendation"), 1 if triage.get("override_applied") else 0,
            (triage.get("override_reason") or {}).get("reason") if isinstance(triage.get("override_reason"), dict) else triage.get("override_reason"),
            json.dumps(triage.get("possible_differentials",[]), ensure_ascii=False),
            json.dumps(triage.get("shap_top",[]), ensure_ascii=False),
            json.dumps(gen, ensure_ascii=False),
            1 if state.get("disclaimer_ack") else 0,
            json.dumps(state, ensure_ascii=False),
          )
        )

# ---------- Detección de rama según motivo/localización ----------
def detect_route(text:str)->str:
    t = normalize(text)
    if any(k in t for k in ["pecho","precordial","aprieta el pecho","dolor de pecho","torac"]): return "pecho"
    if any(k in t for k in ["panza","vientre","abdomen","estómago","estomago","barriga","cólico"]): return "panza"
    if any(k in t for k in ["cabeza","migraña","migraña"]): return "cabeza"
    if any(k in t for k in ["tos","respirar","respir","garganta","falta de aire","ahogo","pecho al respirar"]): return "resp"
    if any(k in t for k in ["orinar","orina","pis","ardor al orinar","cistitis","pipi"]): return "urina"
    if any(k in t for k in ["golpe","caída","caida","accidente","fractura","torcedura","esguince","trauma"]): return "trauma"
    return "general"

# Planes por rama (se definen después del motivo)
COMMON_AFTER_MOTIVO = ["antecedentes","medicacion","alergias","factores_riesgo"]
BRANCH_TAIL = ["sintomas_asoc","alivio","desencadenantes","caracter","intensidad","localizacion","dirigidas","consentimiento"]

def build_plan(route:str)->List[str]:
    # después de datos afiliatorios y motivo
    return COMMON_AFTER_MOTIVO + BRANCH_TAIL

def current_order(state:Dict[str,Any])->List[str]:
    prefix = ["nombre_apellido","edad","domicilio","ocupacion","motivo"]
    plan = state.get("plan") or build_plan("general")
    return prefix + plan

def next_question_text(state:Dict[str,Any])->str:
    step = state["current_step"]
    if step == "dirigidas":
        route = state.get("route","general")
        key = f"dirigidas_{route}"
        return UI_QUESTION.get(key, UI_QUESTION["dirigidas_general"])
    return UI_QUESTION.get(step, "Continuar…")

def ui_meta(state:Dict[str,Any])->Dict[str,Any]:
    order = current_order(state)
    step = state["current_step"]
    idx = order.index(step)
    field_type = "text"; placeholder = "Escribí tu respuesta"
    if step in ("edad","intensidad"):
        field_type = "number"; placeholder = "Ingresar número"
    return {
        "current_step": step,
        "current_label": LABELS.get(step, step),
        "step_index": idx+1, "step_total": len(order),
        "field_type": field_type, "placeholder": placeholder,
        "title":"Ayudanos a orientarte",
        "subtitle":"Te vamos a hacer algunas preguntas rápidas para estimar la urgencia.",
    }

# ---------- NLP muy simple + features ----------
def extract_features(ans:Dict[str,str])->Dict[str,Any]:
    t = normalize(" ".join([
        ans.get("motivo",""), ans.get("sintomas_asoc",""), ans.get("caracter",""),
        ans.get("desencadenantes",""), ans.get("alivio",""), ans.get("dirigidas",""),
        ans.get("localizacion",""), ans.get("factores_riesgo",""), ans.get("antecedentes","")
    ]))
    f = {
        "chest_pain": int(any(k in t for k in ["pecho","precordial"])),
        "dyspnea": int(any(k in t for k in ["falta de aire","ahogo"])),
        "palpitations": int(any(k in t for k in ["latidos fuertes","palpitaciones"])),
        "syncope": int(any(k in t for k in ["desmayo"])),
        "radiation_left_arm": int(any(k in t for k in ["brazo izquierdo","mandíbula"])),
        "diaphoresis": int(any(k in t for k in ["sudor","transpiración fría"])),
        "nausea": int(any(k in t for k in ["náusea","nausea","vómit","vomit"])),
        "altered_consciousness": int(any(k in t for k in ["inconsciente","no responde","confuso"])),
        "focal_deficit_sudden": int(any(k in t for k in ["cara caída","no puede levantar","hablar","balbucea"])),
        "massive_bleeding": int(any(k in t for k in ["sangrado abundante","hemorragia masiva"])),
        "hypotension": int("presión muy baja" in t or "hipotens" in t),
        "hemodynamic_instability": int("shock" in t or "muy pálido" in t),
        "diabetes": int("diabetes" in t),
        "htn": int("presión alta" in t or "hipertens" in t),
    }
    try:
        f["severity_num"] = max(0.0, min(10.0, float(ans.get("intensidad","0") or 0)))
    except: f["severity_num"] = 0.0
    ents = {"comorbidities":[k for k in ["diabetes","htn"] if f.get(k)]}
    return {"features":f, "entities":ents}

_WEIGHTS = {"chest_pain":2.5,"dyspnea":2.0,"radiation_left_arm":1.5,"diaphoresis":1.0,"syncope":1.5,"nausea":0.5,"diabetes":0.5,"htn":0.5}

def hard_overrides(f:Dict[str,Any])->Optional[Dict[str,Any]]:
    if f.get("massive_bleeding"): return {"esi":1,"reason":"Sangrado abundante"}
    if f.get("altered_consciousness") and f.get("focal_deficit_sudden"): return {"esi":1,"reason":"Alteración de conciencia + signos focales"}
    if f.get("chest_pain") and f.get("dyspnea") and (f.get("hypotension") or f.get("hemodynamic_instability")):
        return {"esi":1,"reason":"Dolor en pecho + falta de aire + inestabilidad"}
    return None

def risk_score(f:Dict[str,Any])->float:
    s = sum(_WEIGHTS[k] for k in _WEIGHTS if f.get(k))
    s += 0.1 * float(f.get("severity_num",0))
    return s

def predict_esi(f:Dict[str,Any])->Tuple[int, Optional[Dict[str,Any]], List[Dict[str,Any]]]:
    ov = hard_overrides(f)
    if ov: return 1, ov, [{"feature": ov["reason"], "value": 1, "contrib": 6.0}]
    s = risk_score(f)
    if   s >= 6.5: esi = 2
    elif s >= 3.5: esi = 3
    elif s >= 1.5: esi = 4
    else:          esi = 5
    shap = [{"feature":k,"value":1,"contrib":(_WEIGHTS.get(k,0) or (0.1 if k=="severity_num" else 0))}
            for k in f if f.get(k)]
    shap.sort(key=lambda x:-x["contrib"])
    return esi, None, shap[:8]

def llm_generate_summary(triage:Dict[str,Any], state:Dict[str,Any])->Dict[str,Any]:
    esi = triage["predicted_ESI"]
    if esi in (1,2):
        studies=["ECG","SatO2 continua","Troponinas","Monitoreo"]; risks=["Posible evento cardiovascular agudo"]
    elif esi==3:
        studies=["Laboratorio básico","ECG si dolor en pecho","Rx tórax si falta de aire"]; risks=["Evaluación en guardia"]
    else:
        studies=["Control ambulatorio"]; risks=["Bajo riesgo aparente"]
    hce=f"Motivo: {state.get('answers',{}).get('motivo','')}. ESI {esi} ({triage.get('mapped_manchester')})."
    return {"suggested_level":f"ESI {esi} ({triage.get('mapped_manchester')})","initial_studies":studies,"risks":risks,
            "hce_text":hce,"disclaimer":"Recomendación asistida por IA; no reemplaza evaluación médica ni indica dosis."}

# ---------- API ----------
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
        sid=gen_session_id()
        state={"session_id":sid,"created_at":now_iso(),"answers":{},"current_step":"nombre_apellido",
               "route":"general","plan":None,"disclaimer_ack":False,"version":APP_VERSION}
        SESSIONS[sid]=state
        return {"session_id":sid,"next_question":next_question_text(state),"completed":False,"ui":ui_meta(state)}

    # Recuperar estado
    state=SESSIONS.get(session_id)
    if not state: raise HTTPException(404,"Sesión no encontrada")
    step=state["current_step"]
    ans=state["answers"]

    # Defaults mínimos para no romper si el input viene vacío en pasos opcionales
    if not user_input:
        defaults={"alergias":"no","antecedentes":"ninguna","medicacion":"ninguna"}
        user_input=defaults.get(step,"")

    # Guardar respuesta
    ans[step]=user_input

    # Armar patient básico
    state["patient"]={"nombre_apellido":ans.get("nombre_apellido"),
                      "edad":ans.get("edad"),"domicilio":ans.get("domicilio"),
                      "ocupacion":ans.get("ocupacion")}

    # Si definió MOTIVO, fijamos la ruta y plan dinámico
    if step=="motivo":
        state["route"]=detect_route(user_input)
        state["plan"]=build_plan(state["route"])

    # Si más adelante indicó localización y la ruta era general, intentamos refinar
    if step=="localizacion" and (state.get("route")=="general"):
        state["route"]=detect_route(user_input)

    # Chief complaint sintético para panel
    cc_parts=[ans.get("motivo",""), ans.get("localizacion",""), ans.get("caracter","")]
    state["chief_complaint"]=" | ".join([p for p in cc_parts if p]).strip()

    # ¿Completó?
    order=current_order(state)
    idx=order.index(step)
    if step=="consentimiento":
        # extraer features + clasificar
        nlp=extract_features(ans); f, ents=nlp["features"], nlp["entities"]
        esi, override, shap_top = predict_esi(f)
        if esi==1:
            diffs=["Emergencia mayor"]; rec="Atención inmediata y monitoreo."
        elif esi==2:
            diffs=["Posible SCA","TEP","Insuficiencia respiratoria"]; rec="Urgencias (ECG/Troponinas/monitoreo)."
        elif esi==3:
            diffs=["Dolor de pecho inespecífico","Reflujo","Costocondritis"]; rec="Evaluación en guardia."
        elif esi==4:
            diffs=["Cefalea tensional","Lumbalgia","Infección leve"]; rec="Atención diferida/ambulatoria."
        else:
            diffs=["Consulta simple"]; rec="Alta con recomendaciones."
        tri={"predicted_ESI":esi,"mapped_manchester":MANCHESTER_MAP.get(esi),
             "possible_differentials":diffs,"recommendation":rec,"shap_top":shap_top,
             "override_applied":bool(override),"override_reason":override}
        gen=llm_generate_summary(tri, state)
        state.update({"derived_features":f,"extracted_entities":ents,"final_triage":tri,
                      "llm_summary":gen,"disclaimer_ack":True})
        save_session_to_db(state)
        return {"session_id":state["session_id"],"completed":True,"triage_result":tri,"generative_summary":gen}

    # Avanzar al siguiente paso del plan
    next_step = order[min(idx+1, len(order)-1)]
    state["current_step"]=next_step
    # Para la clave 'dirigidas' usamos el texto según la ruta
    nq = next_question_text(state)
    return {"session_id":state["session_id"],"next_question":nq,"completed":False,"ui":ui_meta(state)}

# ---------- Panel & export ----------
@app.get("/admin/sessions")
def list_sessions():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory=sqlite3.Row
        rows=con.execute("""
        SELECT session_id,created_at,chief_complaint,predicted_esi,mapped_manchester,
               recommendation,override_applied,override_reason
        FROM sessions ORDER BY datetime(created_at) DESC
        """).fetchall()
        return JSONResponse([dict(r) for r in rows])

@app.get("/admin/sessions/{session_id}")
def session_detail(session_id:str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory=sqlite3.Row
        r=con.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not r: raise HTTPException(404,"not found")
        d=dict(r)
        for k in ["patient_json","extracted_entities_json","derived_features_json","differentials_json","shap_top_json","llm_summary_json","state_json"]:
            if d.get(k):
                try: d[k]=json.loads(d[k])
                except: pass
        return JSONResponse(d)

@app.get("/export/excel")
def export_excel():
    with sqlite3.connect(DB_PATH) as con:
        df=pd.read_sql_query("""
          SELECT session_id as Sesion, created_at as Fecha, chief_complaint as Motivo,
                 predicted_esi as ESI, mapped_manchester as Manchester, recommendation as Recomendacion,
                 override_applied as Override, override_reason as MotivoOverride
          FROM sessions ORDER BY datetime(Fecha) DESC
        """, con)
    buf=BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False, sheet_name="Resumen")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="triage_export.xlsx"'})

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    html="""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
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
    <td>${(x.chief_complaint||'').slice(0,140)}</td>
    <td>${x.predicted_esi ?? ''}</td>
    <td><span class="tag ${x.mapped_manchester||''}">${x.mapped_manchester||''}</span></td>
    <td>${(x.recommendation||'').slice(0,120)}</td>
    <td>${x.override_applied ? 'Sí' : 'No'}</td>`;
  tb.appendChild(tr);
 }
})();
</script>"""
    return HTMLResponse(html)

@app.exception_handler(404)
async def not_found(_, __): return PlainTextResponse("No encontrado", status_code=404)
