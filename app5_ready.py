import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# pealkiri
st.title("ÕIS-i ainete nõustaja")
st.caption("Vali vasakult menüüst filtrid ja kirjelda, mida soovid otsida.")

JOIN_COL = "unique_ID"
CITY_COL = "linn"
SEM_COL  = "semester"
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

UI_TO_NORM = {
    "Tartu": "tartu",
    "Tallinn": "tallinn",
    "Viljandi": "viljandi",
    "Narva": "narva",
    "Pärnu": "parnu",
    "Tõravere": "toravere",
    "Muu": "muu",
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

    use_city = st.checkbox("Kasuta linna filtrit", value=False)
    city_choice = "Kõik"
    if use_city:
        city_choice = st.selectbox(
            "Linn",
            ["Kõik", "Tartu", "Tallinn", "Viljandi", "Narva", "Pärnu", "Tõravere", "Muu"],
            index=0
        )
        st.caption("Märkus: puuduva linna korral eeldame Tartut.")


# 1. alustame
if "messages" not in st.session_state:
    st.session_state.messages = []
# 2. kuvame ajaloo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. kuulame kasutaja sõnumit
if prompt := st.chat_input("Kirjelda, mida soovid otsida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # Filtreerimine enne semantilist otsingut - RAG
            with st.spinner("Otsin sobivaid kursusi..."):

                # 1. Filtreeri andmetabel
                # Kasutame .copy(), et vältida hilisemaid hoiatusi andmete muutmise kohta

                # merge and CREATE city_norm (THIS is where normalize_city is called)
                merged_df = pd.merge(df, embeddings_df, on=JOIN_COL, how="inner").copy()
                merged_df["city_norm"] = merged_df[CITY_COL].apply(normalize_city)

                # build filter mask
                mask = pd.Series(True, index=merged_df.index)

                if semester_choice != "Kõik":
                    mask &= (merged_df[SEM_COL] == semester_choice)

                if int(eap_choice) != 0:
                    mask &= (merged_df[EAP_COL] == int(eap_choice))

                if use_city and city_choice != "Kõik":
                    chosen_norm = UI_TO_NORM[city_choice]
                    mask &= (merged_df["city_norm"] == chosen_norm)

                filtered_df = merged_df[mask].copy()
                
                
                #kontroll (sanity check)
                if filtered_df.empty:
                    st.markdown(
                            "Ei leidnud ühtegi kursust nende filtritega. "
                            "Proovi muuta semestrit/EAP-i või lülita linna filter välja."
                        )
                    st.stop()
                else:
                    # Arvutame sarnasuse ja sorteerime tabeli
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
                    context_text = results_df.to_string(index=False)

                # 3. LLM vastus
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": (
                        "Oled ülikooli kursusenõustaja. Soovita kasutajale kursusi allolevast nimekirjast. "
                        "Vastuses maini kindlasti kursuste nime, ainekoodi, EAP arvu, kevad või sügissemestrit ja kui võimalik, siis toimumise asukohta. \n"
                        f"Kursused:\n\n{context_text}"
                    )
                }
                
                messages_to_send = [system_prompt] + st.session_state.messages
                
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
                    st.session_state.messages.append({"role": "assistant", "content": full_text})

                    if final_usage:
                        token_in = int(final_usage.prompt_tokens or 0)
                        token_out = int(final_usage.completion_tokens or 0)

                        st.session_state.stats["tokens_in"] += token_in
                        st.session_state.stats["tokens_out"] += token_out

                        # your price function (per 1M tokens)
                        st.session_state.stats["cost"] += mudel(token_in, token_out)


                except Exception as e:
                    st.error(f"Viga: {e}")

with st.sidebar:
    st.markdown("---")
    st.markdown("### Sessiooni statistika")
    st.write("Sisendtokenid:", st.session_state.stats["tokens_in"])
    st.write("Tokens out:", st.session_state.stats["tokens_out"])
    st.write("Cost (€):", round(st.session_state.stats["cost"], 6))