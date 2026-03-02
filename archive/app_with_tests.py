import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import csv
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# NB - NOT SURE IF THIS WORKS AT ALL. MOVE OUT OF THIS FOLDER TO RUN

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


# --- TESTIMISE SEADISTUSED / OLEK ---
TEST_CASES_DEFAULT_PATH = "test_cases.csv"
TEST_LOG_PATH = "log/test_results.csv"

if "test_stats" not in st.session_state:
    st.session_state.test_stats = {"passed": 0, "failed": 0, "last_run": None}

if "test_last_results" not in st.session_state:
    st.session_state.test_last_results = []  # list of dicts

def ensure_log_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_test_cases(path: str) -> pd.DataFrame:
    """Loads test cases from CSV.

    Your uploaded CSV has an empty first column, then:
      col 1 = prompt
      col 2 = expected unique_ID list (comma-separated)
    We handle both headered and headerless CSVs robustly.
    """
    try:
        df_tc = pd.read_csv(path, header=None)
        # Expected format: [empty, prompt, expected]
        if df_tc.shape[1] >= 3:
            df_tc = df_tc.iloc[:, [1, 2]].copy()
            df_tc.columns = ["prompt", "expected_ids"]
        elif df_tc.shape[1] == 2:
            df_tc.columns = ["prompt", "expected_ids"]
        else:
            return pd.DataFrame(columns=["prompt", "expected_ids"])
        df_tc["prompt"] = df_tc["prompt"].astype(str).str.strip()
        df_tc["expected_ids"] = df_tc["expected_ids"].astype(str).str.strip()
        df_tc = df_tc[df_tc["prompt"].str.len() > 0].reset_index(drop=True)
        return df_tc
    except Exception:
        # Fallback: try headered read
        try:
            df_tc = pd.read_csv(path)
            if df_tc.shape[1] >= 2:
                # pick last 2 columns
                df_tc = df_tc.iloc[:, -2:].copy()
                df_tc.columns = ["prompt", "expected_ids"]
                df_tc["prompt"] = df_tc["prompt"].astype(str).str.strip()
                df_tc["expected_ids"] = df_tc["expected_ids"].astype(str).str.strip()
                return df_tc[df_tc["prompt"].str.len() > 0].reset_index(drop=True)
        except Exception:
            pass
        return pd.DataFrame(columns=["prompt", "expected_ids"])

def split_expected_ids(s: str) -> list[str]:
    if s is None:
        return []
    # split on comma and/or whitespace-newlines, keep dot/number codes intact
    parts = []
    for p in str(s).replace("\n", ",").split(","):
        p = p.strip()
        if p:
            parts.append(p)
    return parts

def missing_ids_in_response(response: str, expected_ids: list[str]) -> list[str]:
    """Returns expected IDs that do NOT appear as substrings in response (case-insensitive)."""
    resp = (response or "").lower()
    missing = [eid for eid in expected_ids if eid.lower() not in resp]
    return missing

def build_rag_context(user_prompt: str,
                      semester_choice: str,
                      eap_choice: int,
                      use_more: bool,
                      hindamine_choice: str,
                      keel_choice: str,
                      city_choice: str):
    """Runs the same filtering + vector search you use in chat. Returns:
    (current_filters_str, filtered_count, results_df_display, context_text)
    """
    if use_more:
        current_filters_str = f"EAP:{eap_choice}, Sem:{semester_choice}, Hind:{hindamine_choice}, Linn:{city_choice}, Keel:{keel_choice}"
    else:
        current_filters_str = f"EAP:{eap_choice}, Sem:{semester_choice}"

    merged_df = pd.merge(df, embeddings_df, on=JOIN_COL, how="inner").copy()
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
            lambda s: keel_choice.lower() in [x.strip().lower() for x in str(s).split(",") if x.strip()]
        )

    merged_df["city_norm"] = merged_df[CITY_COL].apply(normalize_city)
    if use_more and city_choice != "Kõik":
        chosen_norm = UI_TO_NORM[city_choice]
        mask &= (merged_df["city_norm"] == chosen_norm)

    filtered_df = merged_df[mask].copy()
    filtered_count = len(filtered_df)

    if filtered_df.empty:
        return current_filters_str, filtered_count, pd.DataFrame(), "Sobivaid kursusi ei leitud."

    query_vec = embedder.encode([user_prompt])[0]
    filtered_df["score"] = cosine_similarity([query_vec], np.stack(filtered_df["embedding"]))[0]

    results_N = 5
    results_df = (
        filtered_df
        .sort_values("score", ascending=False)
        .head(results_N)
        .drop(columns=["score", "embedding"], errors="ignore")
    )
    results_df_display = results_df.drop(columns=["embedding"], errors="ignore").copy()

    context_text = results_df.to_string(index=False)
    return current_filters_str, filtered_count, results_df_display, context_text

def ask_llm(user_prompt: str, api_key: str, context_text: str, chat_history_messages: list[dict] | None = None):
    """Non-stream LLM call (used for tests)."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_prompt = {
        "role": "system",
        "content": (
            "Oled ülikooli kursusenõustaja. Soovita kasutajale kursusi allolevast nimekirjast. "
            "Ära vasta mõttepunktidena, vaid pane tekst pigem lõiguna kirja, et säästa ruumi. "
            "Vastuses maini kindlasti kursuste nime esimesena, ainekoodi (unique_ID), EAP arvu, kevad või sügissemestrit, "
            "hindamisviisi, toimumise asukohta. "
            f"Kursused:\n\n{context_text}"
        )
    }

    if chat_history_messages is None:
        messages = [system_prompt, {"role": "user", "content": user_prompt}]
    else:
        messages = [system_prompt] + chat_history_messages

    resp = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=messages,
        stream=False,
    )
    text = resp.choices[0].message.content if resp.choices else ""
    usage = getattr(resp, "usage", None)
    return text, usage, system_prompt["content"]

def append_test_log(rows: list[dict]) -> None:
    ensure_log_dir(TEST_LOG_PATH)
    file_exists = os.path.isfile(TEST_LOG_PATH)
    import csv as _csv
    with open(TEST_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "prompt", "expected_ids", "passed", "missing_ids",
                "filters", "filtered_count", "retrieved_ids", "response"
            ],
        )
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

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

# --- Viimase testijooksu kokkuvõte (kui olemas) ---
if st.session_state.get("test_last_results"):
    with st.expander("Viimase testijooksu tulemused"):
        df_res = pd.DataFrame(st.session_state.test_last_results)
        # show only key columns by default
        st.dataframe(df_res[["passed", "prompt", "expected_ids", "missing_ids", "retrieved_ids_top5"]], hide_index=True)
        st.caption("Detailne LLM vastus on veerus 'response'. Kõik vastused on lisaks logitud faili log/test_results.csv.")

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

    st.markdown("---")
    st.subheader("Testimine (test_cases.csv)")
    tc_upload = st.file_uploader("Lae test_cases.csv (valikuline)", type=["csv"])
    run_tests_btn = st.button("Käivita testid")
    reset_tests_btn = st.button("Nulli testi loendur")


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


# --- Testi nupud ---
if reset_tests_btn:
    st.session_state.test_stats = {"passed": 0, "failed": 0, "last_run": None}
    st.session_state.test_last_results = []
    st.toast("Testi loendur nullitud.")

if run_tests_btn:
    if not api_key:
        st.error("Testide käivitamiseks sisesta OpenRouter API Key.")
    else:
        # load test cases (uploaded file wins)
        if tc_upload is not None:
            tc_path = tc_upload
            df_tc = pd.read_csv(tc_path, header=None)
            # Normalize to expected columns
            if df_tc.shape[1] >= 3:
                df_tc = df_tc.iloc[:, [1, 2]].copy()
                df_tc.columns = ["prompt", "expected_ids"]
            else:
                df_tc.columns = ["prompt", "expected_ids"]
        else:
            df_tc = load_test_cases(TEST_CASES_DEFAULT_PATH)

        if df_tc.empty:
            st.error("Test_case'e ei leitud. Veendu, et test_cases.csv on olemas või lae see üles.")
        else:
            passed = 0
            failed = 0
            results = []
            log_rows = []
            progress = st.progress(0)
            status = st.empty()

            total = len(df_tc)
            for idx_tc, row in df_tc.iterrows():
                p = str(row["prompt"]).strip()
                exp_str = str(row["expected_ids"]).strip()
                expected = split_expected_ids(exp_str)

                status.write(f"Test {idx_tc+1}/{total}: {p[:80]}")
                try:
                    filters_str, filtered_count, ctx_df, ctx_text = build_rag_context(
                        p, semester_choice, int(eap_choice), use_more,
                        hindamine_choice, keel_choice, city_choice
                    )
                    response_text, usage, system_prompt_text = ask_llm(p, api_key, ctx_text)

                    missing = missing_ids_in_response(response_text, expected)
                    ok = len(missing) == 0

                    if ok:
                        passed += 1
                    else:
                        failed += 1

                    retrieved_ids = ctx_df["unique_ID"].tolist() if (ctx_df is not None and not ctx_df.empty and "unique_ID" in ctx_df.columns) else []

                    results.append({
                        "prompt": p,
                        "expected_ids": exp_str,
                        "passed": ok,
                        "missing_ids": ", ".join(missing),
                        "retrieved_ids_top5": ", ".join(retrieved_ids),
                        "response": response_text
                    })

                    log_rows.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": p,
                        "expected_ids": exp_str,
                        "passed": ok,
                        "missing_ids": ", ".join(missing),
                        "filters": filters_str,
                        "filtered_count": filtered_count,
                        "retrieved_ids": ", ".join(retrieved_ids),
                        "response": response_text
                    })

                except Exception as e:
                    failed += 1
                    results.append({
                        "prompt": p,
                        "expected_ids": exp_str,
                        "passed": False,
                        "missing_ids": "ERROR",
                        "retrieved_ids_top5": "",
                        "response": f"ERROR: {e}"
                    })
                    log_rows.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": p,
                        "expected_ids": exp_str,
                        "passed": False,
                        "missing_ids": "ERROR",
                        "filters": "",
                        "filtered_count": 0,
                        "retrieved_ids": "",
                        "response": f"ERROR: {e}"
                    })

                progress.progress(int((idx_tc+1) / total * 100))

            append_test_log(log_rows)
            st.session_state.test_stats = {
                "passed": passed,
                "failed": failed,
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.test_last_results = results
            st.success(f"Testid valmis: {passed} läbis, {failed} ebaõnnestus. Logi: {TEST_LOG_PATH}")



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

# --- FIKSEERITUD LOENDUR (vasakul all) ---
_passed = st.session_state.test_stats.get("passed", 0)
_failed = st.session_state.test_stats.get("failed", 0)
_last = st.session_state.test_stats.get("last_run", "—")

st.markdown(
    f"""
    <style>
    .test-counter {{
        position: fixed;
        left: 16px;
        bottom: 16px;
        z-index: 1000;
        background: rgba(0,0,0,0.65);
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.25;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(6px);
        max-width: 260px;
    }}
    </style>
    <div class="test-counter">
      <div><b>Testid</b></div>
      <div>Läbinud: <b>{_passed}</b></div>
      <div>Ebaõnnestunud: <b>{_failed}</b></div>
      <div style="opacity:0.85;">Viimane jooks: {_last}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
