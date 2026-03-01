import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- TAGASISIDE LOGIMISE FUNKTSIOON ---
# Lisasime siia 'context_names', et aine nimed läheksid ka otse logisse
def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category):
    file_path = 'tagasiside_log.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])

# --- MUDELITE JA ANDMETE LAADIMINE ---
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("../andmed/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("../andmed/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

# Pealkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 6")
st.caption("RAG süsteem koos kapotialuse analüüsi ja tagasiside logimisega.")

# --- KÜLGRIBA JA FILTRID ---
with st.sidebar:
    st.header("⚙️ Seaded ja filtrid")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.divider()
    
    st.subheader("Filtreeri kursusi")
    max_eap = float(df['eap'].max()) if 'eap' in df.columns else 60.0
    eap_range = st.slider("EAP maht", 0.0, max_eap, (0.0, max_eap), step=1.0)
    semester_opts = st.multiselect("Semester", ["kevad", "sügis"])
    hindamis_opts = st.multiselect("Hindamisviis", ["Eristav", "Eristamata"])
    linn_opts = st.multiselect("Linn", ["Tartu", "Tallinn", "Narva", "Pärnu", "Viljandi", "Tõravere"])
    aste_opts = st.multiselect("Õppeaste", ["bakalaureuse", "magistri", "doktori"])
    veeb_opts = st.multiselect("Õppevorm", ["põimõpe", "lähiõpe", "veebiõpe"])
    no_prereqs = st.checkbox("Ainult ilma eeldusaineteta kursused")

# --- VESTLUSE LOGIIKA JA AJALUGU ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Kuvame ajaloo koos kapotialuse info ja tagasiside vormidega
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Lisame debug info ja tagasiside ainult assistendi sõnumitele, millel on vajalikud andmed
        if message["role"] == "assistant" and "debug_info" in message:
            debug = message["debug_info"]
            
            # 1. Kapoti all (RAG andmed JA süsteemiviip)
            with st.expander("🔍 Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
                st.write(f"Filtrid jätsid andmestikku alles **{debug.get('filtered_count', 0)}** kursust.")
                
                st.write("**RAG otsingu tulemus (Top 5 leitud kursust):**")
                if not debug.get('context_df').empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score',]
                    cols_to_show = [c for c in display_cols if c in debug.get('context_df').columns]
                    st.dataframe(debug.get('context_df')[cols_to_show], hide_index=True)
                else:
                    st.warning("Ühtegi kursust ei leitud (kas filtrid olid liiga karmid või andmestik tühi).")
                
                # UNIKAALNE KEY LISATUD SIIA
                st.text_area(
                    "LLM-ile saadetud täpne prompt:", 
                    debug.get('system_prompt', ''), 
                    height=150, 
                    disabled=True, 
                    key=f"prompt_area_{i}"
                )
            
            # 2. Tagasiside kogumine
            with st.expander("📝 Hinda vastust (Salvestab logisse)"):
                with st.form(key=f"feedback_form_{i}"):
                    # UNIKAALSED VÕTMED LISATUD SIIA
                    rating = st.radio("Hinnang vastusele:", ["👍 Hea", "👎 Halb"], horizontal=True, key=f"rating_{i}")
                    kato = st.selectbox(
                        "Kui vastus oli halb, siis mis läks valesti?", 
                        ["", "Filtrid olid liiga karmid/valed", "Otsing leidis valed ained (RAG viga)", "LLM hallutsineeris/vastas valesti"],
                        key=f"kato_{i}"
                    )
                    if st.form_submit_button("Salvesta hinnang"):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Võtame lisaks ID-dele välja ka aine nimed logi jaoks
                        ctx_ids = debug.get('context_df')['unique_ID'].tolist() if not debug.get('context_df').empty else []
                        ctx_names = debug.get('context_df')['aine_nimetus_est'].tolist() if (not debug.get('context_df').empty and 'aine_nimetus_est' in debug.get('context_df').columns) else []
                        
                        log_feedback(ts, debug.get('user_prompt', ''), debug.get('filters', ''), ctx_ids, ctx_names, message["content"], rating, kato)
                        st.success("Tagasiside salvestatud tagasiside_log.csv faili!")

# --- KASUTAJA PÄRINGU TÖÖTLEMINE ---
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    current_filters_str = f"EAP:{eap_range}, Sem:{semester_opts}, Hind:{hindamis_opts}, Linn:{linn_opts}, Aste:{aste_opts}, Veeb:{veeb_opts}, Eeldusaineteta:{no_prereqs}"
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = pd.Series(True, index=merged_df.index)
                
                # Filtrite rakendamine
                mask &= (merged_df['eap'] >= eap_range[0]) & (merged_df['eap'] <= eap_range[1])
                if semester_opts: mask &= merged_df['semester'].isin(semester_opts)
                if hindamis_opts:
                    hind_map = {"Eristav": "Eristav (A, B, C, D, E, F, mi)", "Eristamata": "Eristamata (arv, m.arv, mi)"}
                    mask &= merged_df['hindamisviis'].isin([hind_map[h] for h in hindamis_opts])
                if linn_opts:
                    linn_mask = pd.Series(False, index=merged_df.index)
                    if "Tartu" in linn_opts: linn_mask |= merged_df['linn'].isin(["Tartu linn", "Tartu"]) | merged_df['linn'].isna()
                    if "Narva" in linn_opts: linn_mask |= (merged_df['linn'] == "Narva linn")
                    if "Viljandi" in linn_opts: linn_mask |= (merged_df['linn'] == "Viljandi linn")
                    if "Pärnu" in linn_opts: linn_mask |= (merged_df['linn'] == "Pärnu linn")
                    if "Tõravere" in linn_opts: linn_mask |= (merged_df['linn'] == "Tõravere alevik")
                    if "Tallinn" in linn_opts: linn_mask |= (merged_df['linn'] == "Tallinn")
                    mask &= linn_mask
                if aste_opts:
                    pattern = '|'.join(aste_opts)
                    mask &= merged_df['oppeaste'].str.contains(pattern, case=False, na=False)
                if veeb_opts: mask &= merged_df['veebiope'].isin(veeb_opts)
                if no_prereqs: mask &= merged_df['eeldusained'].isna()
                
                filtered_df = merged_df[mask].copy()
                filtered_count = len(filtered_df)
                
                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta valitud filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                    results_df_display = pd.DataFrame()
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df['score'] = cosine_similarity([query_vec], np.stack(filtered_df['embedding']))[0]
                    
                    results_df = filtered_df.sort_values('score', ascending=False).head(5)
                    results_df_display = results_df.drop(columns=['embedding'], errors='ignore').copy()
                    
                    context_text = results_df.drop(columns=['score', 'embedding'], errors='ignore').to_string()

                # --- LLM VASTUS ---
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": f"Oled nõustaja. Järgi kasutaja palveid ning kasuta vastamiseks ainult järgmisi kursusi:\n\n{context_text}"
                }
                
                messages_to_send = [system_prompt] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "debug_info" not in m]
                
                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "debug_info": {
                            "user_prompt": prompt,
                            "filters": current_filters_str,
                            "filtered_count": filtered_count,
                            "context_df": results_df_display,
                            "system_prompt": system_prompt["content"]
                        }
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Viga: {e}")