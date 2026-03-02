import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import csv
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

JOIN_COL = "unique_ID"
CITY_COL = "linn"
SEM_COL  = "semester"
HIN_COL = "hindamisviis"
KEEL_COL = "keel"
EAP_COL  = "eap"

CITY_MAP = {
    "tartu linn": "tartu",
    "tartu": "tartu",
    "tallinn": "tallinn",
    "viljandi linn": "viljandi",
    "narva linn": "narva",
    "pärnu linn": "parnu",
    "tõravere alevik": "toravere",
}

HINDAMINE_MAP = {
    "eristav (a, b, c, d, e, f, mi)" : "eristav", 
    "eristamata (arv, m.arv, mi)" : "eristamata", 
    "kaitsmine" : "eristav"
}

UI_TO_NORM = {
    "Tartu": "tartu",
    "Tallinn": "tallinn",
    "Viljandi": "viljandi",
    "Narva": "narva",
    "Pärnu": "parnu",
    "Tõravere": "toravere",
    "Muu": "muu",
    "Eristav" : "eristav",
    "Eristamata" : "eristamata"
}

if "stats" not in st.session_state:
    st.session_state.stats = {
        "tokens_in": 0,
        "tokens_out": 0,
        "cost": 0.0,
    }

def mudel(token_in, token_out):
    mudel_in_price = 0.04
    mudel_out_price = 0.15
    return (token_in * mudel_in_price / 1_000_000 + token_out * mudel_out_price / 1_000_000)

def normalize_city(x) -> str:
    """Normalize raw city strings into a small set of canonical values.
    Your rule: missing/empty city is assumed to be Tartu.
    """
    if pd.isna(x) or str(x).strip() == "":
        return "tartu"
    key = str(x).strip().lower()
    return CITY_MAP.get(key, "muu")

def normalize_hindamine(x) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return "muu"  # or "puudub"
    key = str(x).strip().lower()
    return HINDAMINE_MAP.get(key, "muu")

# --- TAGASISIDE LOGIMISE FUNKTSIOON ---
# Lisasime siia 'context_names', et aine nimed läheksid ka otse logisse
def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category):
    file_path = 'log/tagasiside_log.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])

# pealkiri
st.title("ÕIS-i ainete nõustaja")
st.caption("Vali vasakult menüüst filtrid ja kirjelda, mida soovid otsida.")

# embed mudel, täisandmestik ja vektorandmebaas läheb cache'i
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("data/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("data/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df
embedder, df, embeddings_df = get_models()

# külgriba
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.subheader("Filtrid")
    semester_choice = st.selectbox("Semester", ["Kõik", "kevad", "sügis"], index=0)

    eap_choice = st.number_input("EAP", min_value=0, max_value=60, value=0, step=1)

    use_more = st.checkbox("Kasuta muusi filreid", value=False)
    hindamine_choice = "Kõik"
    keel_choice = "Kõik"
    city_choice = "Kõik"

    if use_more:
        hindamine_choice = st.selectbox("Hindamine", ["Kõik", "Eristav", "Eristamata"])
        keel_choice = st.selectbox("keel", ["Kõik", "Eesti keel", "Inglise keel", "Vene keel", "Saksa keel", "Prantsuse keel", "Hispaania keel", "Jaapani keel", "Korea keel"])
        city_choice = st.selectbox("Linn", ["Kõik", "Tartu", "Tallinn", "Viljandi", "Narva", "Pärnu", "Tõravere", "Muu"],index=0)


# 1. alustame
if "messages" not in st.session_state:
    st.session_state.messages = []

# kuvame ajaloo koos kapotialuse info ja tagasiside vormidega
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Lisame debug info ja tagasiside ainult assistendi sõnumitele, millel on vajalikud andmed
        if message["role"] == "assistant" and "debug_info" in message:
            debug = message["debug_info"]
            
            # 1. Kapoti all (RAG andmed JA süsteemiviip)
            with st.expander("Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
                st.write(f"Filtrid jätsid andmestikku alles **{debug.get('filtered_count', 0)}** kursust.")
                
                st.write("**RAG otsingu tulemus (Top 5 leitud kursust):**")
                if not debug.get('context_df').empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score', "linn", "keel"]
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
            with st.expander("Hinda vastust (Salvestab logisse)"):
                with st.form(key=f"feedback_form_{i}"):
                    # UNIKAALSED VÕTMED LISATUD SIIA
                    rating = st.radio("Hinnang vastusele:", ["Hea", "Halb"], horizontal=True, key=f"rating_{i}")
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

# 3. kuulame kasutaja sõnumit
if prompt := st.chat_input("Kirjelda, mida soovid otsida..."):
    if use_more:
        current_filters_str = f"EAP:{eap_choice}, Sem:{semester_choice}, Hind:{hindamine_choice}, Linn:{city_choice}, Keel:{keel_choice}"
    else:
        current_filters_str = f"EAP:{eap_choice}, Sem:{semester_choice}"


    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # Filtreerimine enne semantilist otsingut 
            with st.spinner("Otsin sobivaid kursusi..."):

                # 1. Filtreeri andmetabel - 1. metaandmete filtreerimine
                # Kasutame .copy(), et vältida hilisemaid hoiatusi andmete muutmise kohta

                # merge and CREATE city_norm (THIS is where normalize_city is called)
                merged_df = pd.merge(df, embeddings_df, on=JOIN_COL, how="inner").copy()

                # build filter mask
                mask = pd.Series(True, index=merged_df.index)

                if semester_choice != "Kõik":
                    mask &= (merged_df[SEM_COL] == semester_choice)

                if int(eap_choice) != 0:
                    mask &= (merged_df[EAP_COL] == int(eap_choice))

                merged_df["hindamine_norm"] = merged_df[HIN_COL].apply(normalize_hindamine)
                if use_more and hindamine_choice != "Kõik":
                    chosen_norm = UI_TO_NORM[hindamine_choice]
                    mask &= (merged_df["hindamine_norm"] == chosen_norm)
                    
                if use_more and keel_choice != "Kõik":
                    mask &= merged_df[KEEL_COL].fillna("eesti keel").apply(
                        lambda s: keel_choice.lower() in [x.strip().lower() for x in s.split(",") if x.strip()]
                    )
                
                merged_df["city_norm"] = merged_df[CITY_COL].apply(normalize_city)
                if use_more and city_choice != "Kõik":
                    chosen_norm = UI_TO_NORM[city_choice]
                    mask &= (merged_df["city_norm"] == chosen_norm)

                filtered_df = merged_df[mask].copy()
                filtered_count = len(filtered_df)
                
                #kontroll (sanity check)
                if filtered_df.empty:
                    st.markdown(
                            "Ei leidnud ühtegi kursust nende filtritega. "
                            "Proovi muuta semestrit/EAP-i või lülita linna filter välja."
                        )
                    context_text = "Sobivaid kursusi ei leitud."
                    results_df_display = pd.DataFrame()
                    st.stop()
                else:
                    # Arvutame sarnasuse ja sorteerime tabeli - 2. RAG vectors
                    query_vec = embedder.encode([prompt])[0]
                    # np.stack muudab vektorite seeria (Series of arrays) 2D maatriksiks
                    filtered_df['score'] = cosine_similarity([query_vec], np.stack(filtered_df['embedding']))[0]
                    
                    # Sorteerime skoori alusel (suurem on parem) ja võtame 5 parimat
                    results_N = 5
                    results_df = (
                        filtered_df
                        .sort_values("score", ascending=False)
                        .head(results_N)
                        .drop(columns=["score", "embedding"], errors="ignore")
                    )
                    results_df_display = results_df.drop(columns=['embedding'], errors='ignore').copy()
                    context_text = results_df.to_string(index=False)

                # 3. LLM vastus
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": (
                        "Oled ülikooli kursusenõustaja. Soovita kasutajale kursusi allolevast nimekirjast. "                        
                        "Ära vasta mõttepunktidena, vaid pane tekst pigem lõiguna kirja, et säästa ruumi. "
                        "Vastuses maini kindlasti kursuste nime esimesena, ainekoodi, EAP arvu, kevad või sügissemestrit, hindamisviisi, toimumise asukohta."
                        f"Kursused:\n\n{context_text}"
                    )
                }
                
                messages_to_send = [system_prompt] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "debug_info" not in m]
                
                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True,
                        stream_options={"include_usage": True},
                    )

                    # tokenite jaoks
                    placeholder = st.empty()
                    full_text = ""
                    final_usage = None

                    for chunk in stream:
                        # text deltas
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_text += chunk.choices[0].delta.content
                            placeholder.markdown(full_text)

                        # usage appears on the final extra chunk
                        if getattr(chunk, "usage", None):
                            final_usage = chunk.usage

                    # store assistant message
                    #st.session_state.messages.append({"role": "assistant", "content": full_text})

                    if final_usage:
                        token_in = int(final_usage.prompt_tokens or 0)
                        token_out = int(final_usage.completion_tokens or 0)

                        st.session_state.stats["tokens_in"] += token_in
                        st.session_state.stats["tokens_out"] += token_out

                        # your price function (per 1M tokens)
                        st.session_state.stats["cost"] += mudel(token_in, token_out)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_text,
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

with st.sidebar:
    st.markdown("---")
    st.markdown("### Sessiooni statistika")
    st.write("Sisendtokenid:", st.session_state.stats["tokens_in"])
    st.write("Tokens out:", st.session_state.stats["tokens_out"])
    st.write("Cost (€):", round(st.session_state.stats["cost"], 6))