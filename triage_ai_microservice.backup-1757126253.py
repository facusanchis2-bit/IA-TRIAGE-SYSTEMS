from __future__ import annotations

import hashlib, json, os, sqlite3
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

APP_VERSION = "0.10.0-p0-vitals-filters-auth"
DB_PATH = Path(__file__).with_name("triage_sessions.db")
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ------------------- Auth básica -------------------
security = HTTPBasic()
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")

def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok = (credentials.username == ADMIN_USER and credentials.password == ADMIN_PASS)
    if not ok:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# ------------------- Texto preguntas (lenguaje cotidiano) -------------------
UI_QUESTION = {
    "nombre_apellido": "¿Cuál es tu nombre y apellido?",
    "edad": "¿Qué edad tenés? (en años)",
    "domicilio": "¿En qué ciudad o localidad vivís?",
    "ocupacion": "¿Trabajás? ¿En qué?",
    "motivo": "Contanos con tus palabras, ¿cuál es el motivo principal de consulta?",
    # vitales (opcionales)
    "fc": "Si podés, decinos tu frecuencia cardíaca (latidos por minuto). Si no sabés, dejalo vacío.",
    "pas": "Si mediste la presión: ¿cuál fue la sistólica (número grande, mmHg)? Si no sabés, dejalo vacío.",
    "sat02": "¿Cuál es tu saturación de oxígeno (SatO₂ %), si la mediste? Si no sabés, dejalo vacío.",
    "temp": "¿Tenés temperatura corporal medida? (en °C). Si no, dejalo vacío.",
    # base
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
    # "dirigidas" según rama
    "consentimiento": "Escribí 'sí' para confirmar que leíste la recomendación asistida (no reemplaza la evaluación médica).",
}

LABELS = {
    "nombre_apellido":"nombre y apellido","edad":"edad","domicilio":"domicilio","ocupacion":"ocupación",
    "motivo":"motivo de consulta",
    "fc":"frecuencia cardíaca","pas":"presión sistólica","sat02":"saturación O₂","temp":"temperatura (°C)",
    "antecedentes":"enfermedades previas","medicacion":"medicación",
    "alergias":"alergias","factores_riesgo":"factores de riesgo","sintomas_asoc":"síntomas asociados",
    "alivio":"qué lo alivia","desencadenantes":"qué lo empeora","caracter":"cómo lo describís",
    "intensidad":"intensidad (0–10)","localizacion":"dónde lo sentís","dirigidas":"preguntas dirigidas",
    "consentimiento":"consentimiento"
}

MANCHESTER_MAP = {1:"Red",2:"Orange",3:"Yellow",4:"Green",5:"Blue"}

# ------------------- App & CORS -------------------
app = FastAPI(title="Triage Inteligente", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------------- DB + migración idempotente -------------------
def init_db():
    def col_exists(con, table, col):
        cur = con.execute(f"PRAGMA table_info({table})")
        return any(r[1] == col for r in cur.fetchall())

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

        needed_cols = [
            ("created_at", "TEXT"),
            ("chief_complaint", "TEXT"),
            ("override_reason", "TEXT"),
        ]
        for name, typ in needed_cols:
            if not col_exists(con, "sessions", name):
                con.execute(f"ALTER TABLE sessions ADD COLUMN {name} {typ}")
        con.commit()

@app.on_event("startup")
async def _startup(): init_db()

def save_session_to_db(state: Dict[str, Any]) -> None:
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
        ( state["session_id"], state["created_at"],
          json.dumps(state.get("patient",{}), ensure_ascii=False),
          state.get("chief_complaint",""),
          json.dumps(state.get("extracted_entities",{}), ensure_ascii=False),
          json.dumps(state.get("derived_features",{}), ensure_ascii=False),
          state.get("final_triage",{}).get("predicted_ESI"),
          state.get("final_triage",{}).get("mapped_manchester"),
          state.get("final_triage",{}).get("recommendation"),
          1 if state.get("final_triage",{}).get("override_applied") else 0,
          (state.get("final_triage",{}).get("override_reason") or {}).get("reason")
            if isinstance(state.get("final_triage",{}).get("override_reason"), dict)
            else state.get("final_triage",{}).get("override_reason"),
          json.dumps(state.get("final_triage",{}).get("possible_differentials",[]), ensure_ascii=False),
          json.dumps(state.get("final_triage",{}).get("shap_top",[]), ensure_ascii=False),
          json.dumps(gen, ensure_ascii=False),
          1 if state.get("disclaimer_ack") else 0,
          json.dumps(state, ensure_ascii=False)
        ))

# ------------------- Helpers -------------------
def normalize(s:str)->str: return (s or "").lower().strip()
def now_iso()->str: return datetime.utcnow().isoformat(timespec="seconds")
def gen_session_id()->str:
    raw=f"{datetime.utcnow().isoformat()}-{APP_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
def append_timeline(st,ev,data=None): st.setdefault("timeline",[]).append({"ts":now_iso(),"event":ev,"data":data})

# ------------------- Ramas dinámicas -------------------
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
    if path=="pecho":
        return "Para el dolor en el pecho: ¿cuándo empezó? ¿Se va al brazo izquierdo o mandíbula? ¿Te falta el aire? ¿Sudor frío? ¿Náuseas?"
    if path=="abdomen":
        return "Para dolor de panza: ¿en qué zona? ¿Fiebre? ¿Vómitos? ¿Diarrea o estreñimiento? ¿Embarazo posible?"
    if path=="cabeza":
        return "Dolor de cabeza: ¿empezó de golpe? ¿Rigidez de cuello? ¿Dificultad para hablar o mover? ¿Molesta la luz?"
    if path=="respiratorio":
        return "Respiratorio: ¿tos con flema? ¿Fiebre? ¿Te falta el aire al caminar o en reposo? ¿Dolor al respirar?"
    if path=="urinario":
        return "Al orinar: ¿ardor? ¿Aumentaron las ganas? ¿Sangre en la orina? ¿Dolor en cintura?"
    if path=="trauma":
        return "Trauma: ¿cómo fue el golpe? ¿Podés apoyar o mover? ¿Deformidad? ¿Dolor con el movimiento? ¿Pérdida de conocimiento?"
    if path=="lumbalgia":
        return "Dolor lumbar: ¿desde cuándo? ¿Se va a la pierna? ¿Hormigueo/debilidad? ¿Incontinencia? ¿Fiebre o pérdida de peso?"
    if path=="piel":
        return "Piel/erupción: ¿desde cuándo? ¿Pica? ¿Se extiende? ¿Fiebre? ¿Con dolor o ampollas?"
    if path=="mareo":
        return "Mareo/vértigo: ¿súbito o gradual? ¿Empeora con movimientos? ¿Zumbidos, vómitos, visión doble, debilidad o dificultad para hablar?"
    return None

def build_steps(path:str)->List[str]:
    return [
        "nombre_apellido","edad","domicilio","ocupacion",
        "motivo",
        # vitales opcionales
        "fc","pas","sat02","temp",
        # base
        "antecedentes","medicacion","alergias","factores_riesgo",
        "sintomas_asoc","alivio","desencadenantes","caracter","intensidad","localizacion",
        "dirigidas",  # solo si la rama tiene followups
        "consentimiento",
    ]

# ------------------- NLP/Features & Clasificación -------------------
def simple_ner_and_features(ans:Dict[str,str])->Dict[str,Any]:
    txt = " ".join(normalize(ans.get(k,"")) for k in [
        "motivo","sintomas_asoc","caracter","localizacion","desencadenantes","alivio","antecedentes","factores_riesgo","dirigidas"
    ])
    feat={
        "chest_pain": int(any(k in txt for k in ["pecho","precordial"])),
        "dyspnea": int(any(k in txt for k in ["falta de aire","ahogo"])),
        "palpitations": int(any(k in txt for k in ["latidos fuertes","palpitaciones"])),
        "syncope": int(any(k in txt for k in ["desmayo","se desmayó"])),
        "radiation_left_arm": int(any(k in txt for k in ["brazo izquierdo","mandíbula"])),
        "diaphoresis": int(any(k in txt for k in ["sudor","transpiración fría"])),
        "nausea": int(any(k in txt for k in ["náusea","nausea","vómit","vomit"])),
        "sudden_onset": int(any(k in txt for k in ["inicio súbito","de repente","brusco"])),

        "altered_consciousness": int(any(k in txt for k in ["inconsciente","no responde","confuso","somnoliento"])),
        "focal_deficit_sudden": int(any(k in txt for k in ["cara caída","hablar mal","debilidad del brazo","no mueve un lado"])),
        "massive_bleeding": int(any(k in txt for k in ["sangrado abundante","hemorragia masiva"])),

        "hypotension": int("presión muy baja" in txt or "hipotens" in txt),
        "hemodynamic_instability": int("shock" in txt or "muy pálido" in txt),

        "diabetes": int("diabetes" in txt),
        "htn": int("presión alta" in txt or "hipertens" in txt),
    }
    # vitales numéricos
    def fnum(key, default=0.0):
        try: return float(str(ans.get(key,"")).strip().replace(",",".")) if str(ans.get(key,"")).strip()!="" else default
        except: return default
    feat["severity_num"]=max(0.0,min(10.0,fnum("intensidad",0.0)))
    feat["hr"]=fnum("fc",0.0)
    feat["sbp"]=fnum("pas",0.0)
    feat["spo2"]=fnum("sat02",0.0)
    feat["temp"]=fnum("temp",0.0)

    entities={
        "symptoms":[k for k,v in feat.items() if v and k in ["chest_pain","dyspnea","palpitations","syncope","radiation_left_arm","diaphoresis","nausea","sudden_onset"]],
        "neuro":[k for k,v in feat.items() if v and k in ["altered_consciousness","focal_deficit_sudden"]],
        "bleeding":["massive_bleeding"] if feat.get("massive_bleeding") else [],
        "comorbidities":[k for k,v in feat.items() if v and k in ["diabetes","htn"]],
        "vitals":{"hr":feat["hr"],"sbp":feat["sbp"],"spo2":feat["spo2"],"temp":feat["temp"]},
    }
    return {"features":feat,"entities":entities}

_WEIGHTS={
    "chest_pain":2.5,"dyspnea":2.0,"radiation_left_arm":1.5,"diaphoresis":1.0,"nausea":0.5,"sudden_onset":1.0,
    "palpitations":0.5,"syncope":1.5,"diabetes":0.5,"htn":0.5,
}

def hard_overrides(feat:Dict[str,Any])->Optional[Dict[str,Any]]:
    # Emergencias obvias
    if feat.get("massive_bleeding"): return {"esi":1,"reason":"Sangrado abundante"}
    if feat.get("altered_consciousness") and feat.get("focal_deficit_sudden"): return {"esi":1,"reason":"Alteración de conciencia + signos focales"}
    if feat.get("chest_pain") and feat.get("dyspnea") and (feat.get("hypotension") or feat.get("hemodynamic_instability")):
        return {"esi":1,"reason":"Dolor de pecho + falta de aire + inestabilidad"}
    # Vitales
    if feat.get("spo2",0) and feat["spo2"] < 90:
        return {"esi":2,"reason":"SatO₂ < 90%"}
    if feat.get("sbp",0) and feat["sbp"] < 90:
        return {"esi":2,"reason":"PAS < 90 mmHg"}
    return None

def risk_score(f:Dict[str,Any])->float:
    s=sum(_WEIGHTS[k] for k in _WEIGHTS if f.get(k))
    s += 0.1*float(f.get("severity_num",0))
    # penalizaciones por vitales alterados (suaves si no disparan override)
    if f.get("spo2",0) and f["spo2"]<93: s += 0.7
    if f.get("temp",0) and f["temp"]>=39.5: s += 0.6
    if f.get("sbp",0) and f["sbp"]<100: s += 0.4
    if f.get("hr",0) and f["hr"]>=120: s += 0.4
    return s

def predict_esi(f:Dict[str,Any])->Tuple[int,Optional[Dict[str,Any]],List[Dict[str,Any]]]:
    ov=hard_overrides(f)
    if ov: return max(1,min(2,ov["esi"])),ov,[{"feature":ov["reason"],"value":1,"contrib":6.0}]
    s=risk_score(f)
    if   s>=6.5: esi=2
    elif s>=3.5: esi=3
    elif s>=1.5: esi=4
    else:        esi=5
    shap=[{"feature":k,"value":1,"contrib":(_WEIGHTS.get(k,0) or (0.1 if k=="severity_num" else 0))}
          for k in f if f.get(k)]
    # agregar vitales a SHAP-like explicativo
    for vk,contrib in [("spo2",0.9),("sbp",0.6),("hr",0.4),("temp",0.5)]:
        if f.get(vk,0):
            shap.append({"feature":vk,"value":f[vk],"contrib":contrib})
    shap.sort(key=lambda x:-x["contrib"])
    return esi,None,shap[:8]

def llm_generate_summary(triage:Dict[str,Any], state:Dict[str,Any])->Dict[str,Any]:
    esi=triage.get("predicted_ESI")
    if esi in (1,2):
        studies=["ECG","SatO2 continua","Troponinas","Monitoreo"]; risks=["Riesgo cardiovascular/respiratorio agudo"]
    elif esi==3:
        studies=["Laboratorio básico","ECG si dolor en pecho","Rx tórax si falta de aire"]; risks=["Evaluación en guardia"]
    else:
        studies=["Control ambulatorio"]; risks=["Bajo riesgo aparente"]
    hce=f"Motivo: {state.get('answers',{}).get('motivo','')}. ESI {esi} ({triage.get('mapped_manchester')})."
    return {"suggested_level":f"ESI {esi} ({triage.get('mapped_manchester')})","initial_studies":studies,"risks":risks,
            "hce_text":hce,"disclaimer":"Recomendación asistida por IA; no reemplaza evaluación médica. Sin dosis."}

# ------------------- UI meta -------------------
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

# ------------------- API -------------------
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

    # Datos paciente
    state["patient"]={"nombre_apellido":ans.get("nombre_apellido"),"edad":ans.get("edad"),
                      "domicilio":ans.get("domicilio"),"ocupacion":ans.get("ocupacion")}
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

    # ¿Completó?
    if step=="consentimiento":
        nlp=simple_ner_and_features(ans)
        features, entities = nlp["features"], nlp["entities"]
        esi, override, shap_top = predict_esi(features)

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

        triage_result={"predicted_ESI":esi,"mapped_manchester":MANCHESTER_MAP.get(esi),
                       "possible_differentials":diffs,"recommendation":rec,
                       "shap_top":shap_top,"override_applied":bool(override),"override_reason":override}
        gen=llm_generate_summary(triage_result,state)

        state["extracted_entities"]=entities; state["derived_features"]=features
        state["final_triage"]=triage_result; state["llm_summary"]=gen; state["disclaimer_ack"]=True
        save_session_to_db(state); append_timeline(state,"completed",{"esi":esi})

        return {"session_id":session_id,"completed":True,"triage_result":triage_result,"generative_summary":gen}

    # Avanzar
    state["idx"] += 1
    # Saltear "dirigidas" si la rama no aporta preguntas
    if state["steps"][state["idx"]]=="dirigidas" and not followups_for_path(state.get("path","generico")):
        state["idx"] += 1
    return {"session_id":session_id,"next_question":next_question_for(state),"completed":False,"ui":ui_meta(state)}

# ------------------- Panel + Filtros + Export (protegidos) -------------------
@app.get("/admin/sessions")
def list_sessions(
    auth: bool = Depends(require_admin),
    esi: Optional[int] = Query(None),
    q: Optional[str] = Query(None, min_length=0),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
):
    where = []
    args: List[Any] = []
    if esi in (1,2,3,4,5):
        where.append("predicted_esi = ?"); args.append(esi)
    if q:
        where.append("(session_id LIKE ? OR chief_complaint LIKE ?)"); args += [f"%{q}%", f"%{q}%"]
    if from_:
        where.append("datetime(created_at) >= datetime(?)"); args.append(from_)
    if to:
        where.append("datetime(created_at) <= datetime(?)"); args.append(to)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory=sqlite3.Row
        rows=con.execute(f"""
          SELECT session_id, created_at, chief_complaint,
                 predicted_esi, mapped_manchester, recommendation,
                 override_applied, override_reason
          FROM sessions
          {clause}
          ORDER BY datetime(created_at) DESC
        """, args).fetchall()
        return JSONResponse([dict(r) for r in rows])

@app.get("/admin/sessions/{session_id}")
def session_detail(session_id:str, auth: bool = Depends(require_admin)):
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
def export_excel(auth: bool = Depends(require_admin)):
    with sqlite3.connect(DB_PATH) as con:
        df=pd.read_sql_query("""
          SELECT session_id as Sesion, created_at as Fecha, chief_complaint as Motivo,
                 predicted_esi as ESI, mapped_manchester as Manchester,
                 recommendation as Recomendacion, override_applied as Override
          FROM sessions ORDER BY datetime(Fecha) DESC
        """, con)
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
</style>
<div class="top">
 <h1 style="margin:0;font-size:20px">Panel Médico</h1>
 <a class="btn" href="/export/excel">Descargar Excel</a>
</div>
<div class="row" style="margin:12px 0;">
 <input id="q" placeholder="Buscar por sesión o motivo..." style="min-width:260px">
 <select id="esi">
   <option value="">ESI (todos)</option>
   <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
 </select>
 <input id="from" type="datetime-local">
 <input id="to" type="datetime-local">
 <button class="btn" id="aplicar">Aplicar</button>
</div>
<table id="t"><thead><tr>
<th>Fecha</th><th>Sesión</th><th>Motivo</th><th>ESI</th><th>Manchester</th><th>Recomendación</th><th>Override</th>
</tr></thead><tbody></tbody></table>
<script>
async function load(){
 const q = document.getElementById('q').value.trim();
 const esi = document.getElementById('esi').value.trim();
 const from = document.getElementById('from').value.replace('T',' ');
 const to = document.getElementById('to').value.replace('T',' ');
 const params = new URLSearchParams();
 if (q) params.set('q', q);
 if (esi) params.set('esi', esi);
 if (from) params.set('from', from);
 if (to) params.set('to', to);
 const r = await fetch('/admin/sessions?' + params.toString(), {headers:{'Accept':'application/json'}});
 const rows = await r.json();
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
}
document.getElementById('aplicar').onclick = load;
load();
</script>
"""
    return HTMLResponse(html)

@app.exception_handler(404)
async def not_found(_, __): return PlainTextResponse("No encontrado", status_code=404)
