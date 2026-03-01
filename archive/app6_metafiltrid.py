import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- MUDELITE JA ANDMETE LAADIMINE ---
# Laeme andmed kohe alguses, et saaksime neid külgribal filtrite jaoks kasutada
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("../data/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("../data/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

# Pealkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 5")
st.caption("RAG süsteem koos interaktiivse eel-filtreerimisega.")

# --- KÜLGRIBA JA FILTRID ---
with st.sidebar:
    st.header("⚙️ Seaded ja filtrid")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.divider()
    
    st.subheader("Filtreeri kursusi")
    st.markdown("Jäta väli tühjaks, kui soovid näha kõiki valikuid.")
    
    # 1. EAP (Vahemik nullist kuni andmestiku maksimumini)
    max_eap = float(df['eap'].max()) if 'eap' in df.columns else 60.0
    eap_range = st.slider("EAP maht", 0.0, max_eap, (0.0, max_eap), step=1.0)
    
    # 2. Semester
    semester_opts = st.multiselect("Semester", ["kevad", "sügis"])
    
    # 3. Hindamisviis
    hindamis_opts = st.multiselect("Hindamisviis", ["Eristav", "Eristamata"])
    
    # 4. Linn (Ilma "linn" ja "alevik" sõnadeta)
    linn_opts = st.multiselect("Linn", ["Tartu", "Tallinn", "Narva", "Pärnu", "Viljandi", "Tõravere"])
    
    # 5. Õppeaste
    aste_opts = st.multiselect("Õppeaste", ["bakalaureuse", "magistri", "doktori"])
    
    # 6. Veebiõpe
    veeb_opts = st.multiselect("Õppevorm", ["põimõpe", "lähiõpe", "veebiõpe"])
    
    # 7. Eeldusained
    no_prereqs = st.checkbox("Ainult ilma eeldusaineteta kursused")

# --- VESTLUSE LOGIIKA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
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
                # Liidame tabelid
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                
                # --- FILTREERIMISE LOGIIKA ---
                # Alustame maskiga, kus kõik on tõene (True)
                mask = pd.Series(True, index=merged_df.index)
                
                # 1. EAP filter
                mask &= (merged_df['eap'] >= eap_range[0]) & (merged_df['eap'] <= eap_range[1])
                
                # 2. Semester
                if semester_opts:
                    mask &= merged_df['semester'].isin(semester_opts)
                    
                # 3. Hindamisviis (Vastendamine tagasi originaalväärtustele)
                if hindamis_opts:
                    hind_map = {
                        "Eristav": "Eristav (A, B, C, D, E, F, mi)",
                        "Eristamata": "Eristamata (arv, m.arv, mi)"
                    }
                    mapped_hind = [hind_map[h] for h in hindamis_opts]
                    mask &= merged_df['hindamisviis'].isin(mapped_hind)
                    
                # 4. Linn (Keerulisem loogika: "Tartu" puhul kaasame ka NA)
                if linn_opts:
                    linn_mask = pd.Series(False, index=merged_df.index)
                    if "Tartu" in linn_opts:
                        linn_mask |= merged_df['linn'].isin(["Tartu linn", "Tartu"]) | merged_df['linn'].isna()
                    if "Narva" in linn_opts:
                        linn_mask |= (merged_df['linn'] == "Narva linn")
                    if "Viljandi" in linn_opts:
                        linn_mask |= (merged_df['linn'] == "Viljandi linn")
                    if "Pärnu" in linn_opts:
                        linn_mask |= (merged_df['linn'] == "Pärnu linn")
                    if "Tõravere" in linn_opts:
                        linn_mask |= (merged_df['linn'] == "Tõravere alevik")
                    if "Tallinn" in linn_opts:
                        linn_mask |= (merged_df['linn'] == "Tallinn")
                    mask &= linn_mask
                    
                # 5. Õppeaste (Otsime stringist, case-insensitive)
                if aste_opts:
                    pattern = '|'.join(aste_opts)
                    mask &= merged_df['oppeaste'].str.contains(pattern, case=False, na=False)
                    
                # 6. Veebiõpe
                if veeb_opts:
                    mask &= merged_df['veebiope'].isin(veeb_opts)
                    
                # 7. Eeldusained
                if no_prereqs:
                    mask &= merged_df['eeldusained'].isna()
                
                # Rakendame maski
                filtered_df = merged_df[mask].copy()
                
                # --- SANITY CHECK & SEMANTILINE OTSING ---
                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta valitud filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df['score'] = cosine_similarity([query_vec], np.stack(filtered_df['embedding']))[0]
                    
                    results_N = 5
                    results_df = filtered_df.sort_values('score', ascending=False).head(results_N)
                    results_df.drop(['score', 'embedding'], axis=1, inplace=True)
                    context_text = results_df.to_string()

                # --- LLM VASTUS ---
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": f"Oled nõustaja. Järgi kasutaja palveid ning kasuta vastamiseks ainult järgmisi kursusi:\n\n{context_text}"
                }
                
                messages_to_send = [system_prompt] + st.session_state.messages
                
                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Viga: {e}")