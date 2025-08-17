# build_conversations_with_rag_groq_with_role_selection.py
# RAG + KG + role selection driven generator for 8-month conversations (hypertension scenario)

import os, re, random
import datetime as dt
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import networkx as nx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("Set GROQ_API_KEY in env before running.")

# ---------- CONFIG ----------
START_DATE = dt.date(2023,1,1) 
# START_DATE_OTHER= dt.date(2016,12,4) 
MONTHS = 8
OUT_CONV = "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/conversations_raw.txt"
CHROMA_DIR = ".chroma_rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PERSONAS
PERSONA = {
    "concierge": "Ruby (Elyx Concierge)",
    "doctor": "Dr. Warren (Elyx Medical)",
    "lifestyle": "Advik (Elyx Lifestyle)",
    "nutrition": "Carla (Elyx Nutrition)",
    "pt": "Rachel (Elyx PT)",
    "lead": "Neel (Elyx Concierge Lead)",
    "member": "Rohan"
}

TOPIC_TO_ROLE = {
    "Diagnostics": "doctor",
    "Nutrition": "nutrition",
    "Exercise": "pt",
    "Stress/HRV": "lifestyle",
    "Interventions": "concierge"
}

DATA_PATHS = {
    "fitbit_daily": "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/data/fitbit/dailyActivity_merged.csv",
    "fitbit_sleep": "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/data/fitbit/sleepDay_merged.csv",
    "fitbit_hr": "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/data/fitbit/heartrate_seconds_merged.csv",
    "cvd": "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/data/cvd/cardio_train.csv",
    "cgm": "Elyx_Aug15-18/Schema/conversation/generate_synthetic_convo/data/cgm/glucose_20-10-2023.csv"
}

# ---------- HELPERS ----------
def fmt_date(dt_obj): return f"{dt_obj.month}/{dt_obj.day}/{str(dt_obj.year)[-2:]}"
def fmt_time_random():
    h = random.randint(7,21); m = random.choice([0,15,30,45])
    ampm = "AM" if h<12 else "PM"; hh = h if 1<=h<=12 else (h-12)
    return f"{hh}:{m:02d} {ampm}"

def fmt_line(idx:int, date:dt.date, time_str:str, sender:str, text:str):
    return f"{idx}.  [{date.month}/{date.day}/{str(date.year)[-2:]}, {time_str}] {sender}: {text}"

# ---------- LOADERS ----------
def load_fitbit_daily():
    if os.path.exists(DATA_PATHS["fitbit_daily"]):
        d = pd.read_csv(DATA_PATHS["fitbit_daily"], parse_dates=["ActivityDate"], low_memory=False)
        if "ActivityDate" in d.columns:
            d = d.rename(columns={"ActivityDate":"date","TotalSteps":"Steps"})
        daily = d.groupby("date", as_index=False)["Steps"].sum()
        print("jj")
       
    else:
        rng = pd.date_range(START_DATE, periods=30*MONTHS, freq="D")
        print("jjtt")
       
        daily = pd.DataFrame({"date": rng, "Steps": np.random.normal(8500,2500,len(rng)).clip(1000,20000).astype(int)})

    if os.path.exists(DATA_PATHS["fitbit_sleep"]):
        s = pd.read_csv(DATA_PATHS["fitbit_sleep"], low_memory=False)
        if "SleepDay" in s.columns:
            # Explicitly parse with known format
            s["SleepDay"] = pd.to_datetime(s["SleepDay"], format="%Y-%m-%d", errors="coerce")
            s = s.rename(columns={"SleepDay":"date","TotalMinutesAsleep":"AsleepMin"})
        sagg = s.groupby("date", as_index=False)["AsleepMin"].sum()
    else:
        sagg = pd.DataFrame({"date": daily["date"], "AsleepMin": np.random.normal(420,60,len(daily)).clip(240,720)})

    if os.path.exists(DATA_PATHS["fitbit_hr"]):
        hr = pd.read_csv(DATA_PATHS["fitbit_hr"], parse_dates=["Time"], low_memory=False)
        hr["date"] = hr["Time"].dt.date
        hrv = hr.groupby("date")["Value"].std().reset_index().rename(columns={"Value":"HRV_ms"})
        rhr = hr.groupby("date")["Value"].min().reset_index().rename(columns={"Value":"RHR_bpm"})
        hr_df = pd.merge(hrv, rhr, on="date", how="outer")
    else:
        hr_df = pd.DataFrame({"date": daily["date"], "HRV_ms": np.random.normal(38,10,len(daily)).clip(12,80), "RHR_bpm": np.random.normal(66,6,len(daily)).round().astype(int)})

    # enforce .dt.date
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    sagg["date"] = pd.to_datetime(sagg["date"]).dt.date
    hr_df["date"] = pd.to_datetime(hr_df["date"]).dt.date

    df = pd.merge(daily, sagg, on="date", how="left")
    df = pd.merge(df, hr_df, on="date", how="left")
    df["Sleep_hours"] = (df["AsleepMin"]/60.0).round(2)
    return df[["date","Steps","Sleep_hours","HRV_ms","RHR_bpm"]]

def load_cgm_summary():
    if os.path.exists(DATA_PATHS["cgm"]):
        # Skip the first 2 metadata rows
        c = pd.read_csv(DATA_PATHS["cgm"], skiprows=2, low_memory=False)

        # Ensure expected columns exist
        if "Device Timestamp" in c.columns and "Historic Glucose mmol/L" in c.columns:
            # Parse timestamp
            c["Device Timestamp"] = pd.to_datetime(c["Device Timestamp"], format="%m/%d/%Y %H:%M", errors="coerce")

            # Use Historic glucose, fallback to Scan glucose if missing
            gcol = "Historic Glucose mmol/L"
            if c[gcol].isna().all() and "Scan Glucose mmol/L" in c.columns:
                gcol = "Scan Glucose mmol/L"

            # Convert mmol/L → mg/dL
            c["glucose_mgdl"] = c[gcol] * 18.0

            # Aggregate daily mean & peak
            c["date"] = c["Device Timestamp"].dt.date
            agg = c.groupby("date").agg(
                mean=("glucose_mgdl", "mean"),
                peak=("glucose_mgdl", "max")
            ).reset_index()
            agg["date"] = pd.to_datetime(agg["date"]).dt.date
            return agg

    # Fallback synthetic data
    rng = pd.date_range(START_DATE, periods=30*MONTHS, freq="D")
    return pd.DataFrame({
        "date": rng.date,
        "mean": np.random.normal(106,10,len(rng)).round(1),
        "peak": np.random.normal(150,25,len(rng)).round(1)
    })


def load_bp_samples():
    if os.path.exists(DATA_PATHS["cvd"]):
        # The dataset is semicolon-delimited (cardio_train.csv format)
        bp = pd.read_csv(DATA_PATHS["cvd"], sep=";", low_memory=False)

        # Return only BP columns
        if "ap_hi" in bp.columns and "ap_lo" in bp.columns:
            return bp[["ap_hi", "ap_lo"]]

    # Fallback: synthetic BP samples
    return pd.DataFrame({
        "ap_hi": np.random.normal(145, 12, 1000).astype(int),
        "ap_lo": np.random.normal(92, 8, 1000).astype(int)
    })


# ---------- KG ----------
def build_kg():
    G = nx.DiGraph()
    for k,v in PERSONA.items(): G.add_node(v, ntype="persona", key=k)
    topics = ["Diagnostics","Nutrition","Exercise","Stress/HRV","Interventions"]
    for t in topics: G.add_node(t, ntype="topic")
    G.add_edge("Dr. Warren (Elyx Medical)","Diagnostics", rel="owns")
    G.add_edge("Carla (Elyx Nutrition)","Nutrition", rel="owns")
    G.add_edge("Rachel (Elyx PT)","Exercise", rel="owns")
    G.add_edge("Advik (Elyx Lifestyle)","Stress/HRV", rel="owns")
    G.add_edge("Ruby (Elyx Concierge)","Interventions", rel="coordinates")
    G.add_edge("Neel (Elyx Concierge Lead)","Interventions", rel="reviews")
    return G

# ---------- RAG ----------
def build_rag_index(wearable_df, cgm_df, bp_df, extra_docs):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try: client.delete_collection("elyx_rag")
    except: pass
    col = client.create_collection("elyx_rag")
    embedder = SentenceTransformer(EMBED_MODEL)

    docs, ids, metas = [], [], []
    for i,d in enumerate(extra_docs):
        docs.append(d); ids.append(f"seed_{i:03d}"); metas.append({"type":"seed"})
    for _,r in wearable_df.iterrows():
        d = str(r["date"])
        docs.append(f"Wearable [{d}]: Steps={int(r['Steps'])}, Sleep={r['Sleep_hours']}h, HRV={r['HRV_ms']}, RHR={r['RHR_bpm']}")
        ids.append(f"wear_{d}"); metas.append({"type":"wearable","date":d})
    for _,r in cgm_df.iterrows():
        d = str(r["date"])
        docs.append(f"CGM [{d}]: mean={r['mean']} mg/dL, peak={r['peak']} mg/dL")
        ids.append(f"cgm_{d}"); metas.append({"type":"cgm","date":d})
    for i,r in bp_df.sample(min(100,len(bp_df)), random_state=1).iterrows():
        docs.append(f"BP sample: systolic={int(r['ap_hi'])}, diastolic={int(r['ap_lo'])}")
        ids.append(f"bp_{i:03d}"); metas.append({"type":"bp"})

    embs = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    col.add(documents=docs, ids=ids, embeddings=embs, metadatas=metas)
    return col, embedder

# ---------- Role selection ----------
def select_roles_for_event(event_type, rag_chunks, KG):
    mapping = {
        "diagnostic_order": ["Diagnostics"],
        "diet_change": ["Nutrition"],
        "supplement_start": ["Nutrition"],
        "exercise_update": ["Exercise","Stress/HRV"],
        "concierge_followup": ["Interventions"]
    }
    topics = mapping.get(event_type, ["Interventions"])
    personas = []
    for t in topics:
        for node in KG.predecessors(t):
            if KG.nodes[node].get("ntype") == "persona" or "(" in node:
                personas.append(node)

    top_ids = [rid for rid,_ in rag_chunks[:3]]
    doc_texts = " ".join([doc for _,doc in rag_chunks[:6]])
    if ("bp" in doc_texts.lower() or any("bp_" in rid for rid in top_ids)):
        if "Dr. Warren (Elyx Medical)" not in personas:
            personas.insert(0,"Dr. Warren (Elyx Medical)")
    if ("glucose" in doc_texts.lower() or any("cgm_" in rid for rid in top_ids)):
        if "Carla (Elyx Nutrition)" not in personas:
            personas.insert(0,"Carla (Elyx Nutrition)")
    if ("wearable" in doc_texts.lower() or "hrv" in doc_texts or any("wear_" in rid for rid in top_ids)):
        if "Advik (Elyx Lifestyle)" not in personas:
            personas.append("Advik (Elyx Lifestyle)")

    seen=set(); final=[]
    for p in personas:
        if p not in seen:
            final.append(p); seen.add(p)
    if not final: final=["Ruby (Elyx Concierge)"]
    return final

# ---------- GROQ ----------
def groq_client(): return Groq(api_key=GROQ_KEY)
def groq_generate(messages, model="llama-3.1-8b-instant", temperature=0.35, max_tokens=600):
    client = groq_client()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content

# ---------- DRIVER ----------
def main():
    random.seed(1); np.random.seed(1)
    wearable = load_fitbit_daily()
    cgm = load_cgm_summary()
    bp_samples = load_bp_samples()
    extra_docs = [
         "Guideline: ACC/AHA thresholds: Stage2 >=140/90 as actionable.",
         "Guideline: Pair carbs with protein/fat to blunt CGM spikes."    ]
    KG = build_kg()
    rag_index, embedder = build_rag_index(wearable, cgm, bp_samples, extra_docs)

    all_lines=[]; idx=1
    end_date = (START_DATE + pd.DateOffset(months=MONTHS)).date() - dt.timedelta(days=1)
    cur = START_DATE
    while cur <= end_date:
        print(wearable["date"])
        print("cur")
        print(cur)

        wrow = wearable[wearable["date"] == cur]
        
        if wrow.empty:
            # Generate synthetic data
            wdict = {
                "Steps": int(np.random.normal(8500, 2500)),
                "Sleep_hours": round(np.random.normal(6.8, 0.9), 2),
                "HRV_ms": round(np.random.normal(38, 8), 1),
                "RHR_bpm": int(np.random.normal(66, 5))
            }
        else:
            w = wrow.iloc[0]
            print(w)

            # Use default/fallback if any value is NaN
            wdict = {
                "Steps": int(w.Steps) if pd.notna(w.Steps) else int(np.random.normal(8500, 2500)),
                "Sleep_hours": float(w.Sleep_hours) if pd.notna(w.Sleep_hours) else round(np.random.normal(6.8, 0.9), 2),
                "HRV_ms": float(w.HRV_ms) if pd.notna(w.HRV_ms) else round(np.random.normal(38, 8), 1),
                "RHR_bpm": int(w.RHR_bpm) if pd.notna(w.RHR_bpm) else int(np.random.normal(66, 5))
            }

        cgm_row = cgm[cgm["date"] == cur]
       
        if cgm_row.empty:
            cdict = {"mean": float(np.random.normal(106,9)), "peak": float(np.random.normal(150,20))}
        else:
            cr = cgm_row.iloc[0]; cdict = {"mean": float(cr["mean"]), "peak": float(cr["peak"])}
        

        events=[]
        if wdict["RHR_bpm"]>=72 or wdict["HRV_ms"]<=28: events.append({"type":"exercise_update","reason":"autonomic stress"})
        if cdict["peak"]>=160: events.append({"type":"diet_change","reason":"CGM spike"})
        if cur.day in [7,21]: events.append({"type":"diagnostic_order","reason":"scheduled panel"})
        if cur.weekday()==0: events.append({"type":"concierge_followup","reason":"weekly check-in"})
        if cur.day==10: events.append({"type":"supplement_start","reason":"nutrition suggestion"})

        for ev in events:
            q = f"Date {cur.isoformat()} event {ev['type']} steps {wdict['Steps']} sleep {wdict['Sleep_hours']} HRV {wdict['HRV_ms']} rhr {wdict['RHR_bpm']} cgm_mean {cdict['mean']} cgm_peak {cdict['peak']}"
            q_emb = embedder.encode([q], normalize_embeddings=True)
            res = rag_index.query(query_embeddings=q_emb, n_results=6)
            rag_chunks = list(zip(res["ids"][0], res["documents"][0]))
            roles = select_roles_for_event(ev["type"], rag_chunks, KG)
            persona_instructions = " / ".join([f"{p}" for p in roles])
            sysmsg = {"role":"system","content": ("You are generating short WhatsApp-style messages limited to the listed personas. "
                "ONLY use these personas: " + persona_instructions + ". "
                "Each reply line MUST follow exact format: '<index>.  [M/D/YY, H:MM AM/PM] Name: message' "
                "Do not introduce new names.")}
            user_prompt = f"""
# DATE: {cur.isoformat()}
# EVENT: {ev['type']} - {ev['reason']}
# DAILY_SUMMARY: steps={wdict['Steps']}, sleep={wdict['Sleep_hours']}, HRV={wdict['HRV_ms']}, RHR={wdict['RHR_bpm']}
# CGM_SUMMARY: mean={cdict['mean']}, peak={cdict['peak']}
# RAG_EVIDENCE_SNIPPETS:
# {chr(10).join([f'[{rid}] {doc[:240]}' for rid,doc in rag_chunks])}

# TASK:
# Generate 1-3 messages that a persona from the allowed list would send right now (be realistic, concise).
# Output ONLY the message lines with times. Use times close to now (random within day).
# """
            messages = [sysmsg, {"role":"user","content":user_prompt}]
            out = groq_generate(messages, model="llama-3.1-8b-instant", temperature=0.3, max_tokens=220)
            for line in out.strip().splitlines():
                line=line.strip()
                m = re.match(r"^(?:\d+\.\s*)?\[?([0-9]{1,2}/[0-9]{1,2}/[0-9]{2})?,?\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM))\]?\s*(.+?):\s*(.+)$", line)
                if m:
                    time_str = m.group(2); sender = m.group(3).strip(); text = m.group(4).strip()
                else:
                    parts = line.split(":",1)
                    if len(parts)==2:
                        sender = parts[0].strip(); text = parts[1].strip(); time_str = fmt_time_random()
                    else: continue
                if sender not in roles:
                    for p in roles:
                        if p.split()[0] in sender: sender=p; break
                    else: sender = roles[0]
                formatted = fmt_line(idx, cur, time_str, sender, text)
                all_lines.append(formatted); idx+=1

        cur += dt.timedelta(days=1)

    with open(OUT_CONV,"w",encoding="utf-8") as f:
        for ln in all_lines: 
            f.write(ln + "\n")
            print(ln)
    print("Wrote", OUT_CONV, "with", len(all_lines), "lines.")

if __name__=="__main__":
    main()
