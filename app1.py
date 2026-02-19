import os
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Gemma Chat", page_icon="🤖")
st.title("Gemma Chat (OpenRouter)")

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Optional “system prompt” workaround for Gemma:
# Since Gemma doesn't support system/developer messages on OpenRouter,
# we prepend instructions to the FIRST user message only.
INSTRUCTIONS = (
    "You are a helpful assistant. Answer clearly and concisely.\n\n"
)

# -----------------------------
# API key (env first, then secrets)
# -----------------------------
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    st.error("Missing OPENROUTER_API_KEY. Put it in .streamlit/secrets.toml or as an env var.")
    st.stop()

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    # IMPORTANT: only 'user' and 'assistant' roles. No 'system'.
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Gemma via OpenRouter. Ask me anything 🙂"}
    ]

# -----------------------------
# Render history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# OpenRouter call
# -----------------------------
def call_openrouter(history):
    # Hard filter: only allow roles Gemma supports
    clean = []
    for m in history:
        role = m.get("role", "")
        if role in ("user", "assistant"):
            clean.append({"role": role, "content": str(m.get("content", ""))})

    # If there is at least one user message, prepend instructions ONLY to the first user msg
    first_user_idx = next((i for i, m in enumerate(clean) if m["role"] == "user"), None)
    if first_user_idx is not None:
        clean[first_user_idx]["content"] = INSTRUCTIONS + clean[first_user_idx]["content"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # These two are recommended by OpenRouter (safe to include)
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Gemma Chat",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": clean,
        "temperature": 0.3,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]

# -----------------------------
# Input
# -----------------------------
if prompt := st.chat_input("Type your message..."):
    # Add & show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        try:
            answer = call_openrouter(st.session_state.messages)
        except Exception as e:
            answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
