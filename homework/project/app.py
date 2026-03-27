import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Data files live next to this script; Streamlit Cloud cwd is often the repo root,
# so relative paths like "campaigns.json" fail and crash before any UI → blank page.
BASE_DIR = Path(__file__).resolve().parent


def _campaign_text(c: dict) -> str:
    return (
        f"{c['title']} {c['brand']} {c['agency']} {c['country']} "
        f"{c['industry']} {c['medium']} {c['description']}"
    )


@st.cache_resource
def load_search_stack():
    """Load campaigns + index. Uses embeddings.npy only if present locally (not in git)."""
    data_path = BASE_DIR / "campaigns.json"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing {data_path.name} — keep scraped data local; copy or symlink next to app.py."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        campaigns = json.load(f)

    corpus = [_campaign_text(c).lower().split() for c in campaigns]
    bm25 = BM25Okapi(corpus)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_path = BASE_DIR / "embeddings.npy"
    if emb_path.is_file():
        embeddings = np.load(emb_path)
    else:
        texts = [_campaign_text(c) for c in campaigns]
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )

    return campaigns, embeddings, bm25, model


def hybrid_search(query, campaigns, bm25, embeddings, model, top_k=5, bm25_weight=0.5, semantic_weight=0.5):
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    query_embedding = model.encode([query])
    semantic_scores = np.dot(embeddings, query_embedding.T).flatten()

    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-9
    )
    semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (
        semantic_scores.max() - semantic_scores.min() + 1e-9
    )

    final_scores = bm25_weight * bm25_scores_norm + semantic_weight * semantic_scores_norm
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {
                "score": round(float(final_scores[idx]), 4),
                "id": campaigns[idx]["id"],
                "title": campaigns[idx]["title"],
                "brand": campaigns[idx]["brand"],
                "industry": campaigns[idx]["industry"],
                "description": campaigns[idx]["description"],
                "url": campaigns[idx]["url"],
            }
        )
    return results


def format_campaigns_as_context(results):
    context = ""
    for i, r in enumerate(results, 1):
        context += f"""
Campaign {i}:
- Title: {r['title']}
- Brand: {r['brand']}
- Industry: {r['industry']}
- Score: {r['score']}
- Description: {r['description']}
- URL: {r['url']}
---
"""
    return context


def get_api_key():
    # Streamlit Cloud: set GOOGLE_API_KEY in App settings → Secrets, or use st.secrets
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY")


def get_google_client():
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


st.title("Marketing Campaign Assistant")

google_client = get_google_client()

if not google_client:
    st.error(
        "Missing **GOOGLE_API_KEY**. Add it under App settings → Secrets in Streamlit Cloud, "
        "or set the environment variable locally."
    )
    st.stop()

try:
    with st.spinner("Loading campaign index and embedding model (first run may take a minute)…"):
        campaigns, embeddings, bm25, model = load_search_stack()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about campaigns...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    results = hybrid_search(user_input, campaigns, bm25, embeddings, model, top_k=10)
    context = format_campaigns_as_context(results)

    system_prompt = f"""You are a marketing campaign expert assistant.
Answer based on the retrieved campaigns below.
Answer questions by providing the campaign name, brand, industry, description, and URL.
If the user asks something unrelated to the campaigns, politely redirect them.

Retrieved campaigns:
{context}"""

    user_message = {"role": "user", "parts": [{"text": user_input}]}
    current_contents = st.session_state.conversation_history + [user_message]

    try:
        response = google_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=current_contents,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": 1000,
            },
        )
        assistant_message = response.text
        st.session_state.conversation_history.append(user_message)
        st.session_state.conversation_history.append(
            {"role": "model", "parts": [{"text": assistant_message}]}
        )
        with st.chat_message("assistant"):
            st.write(assistant_message)
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    except Exception as e:
        st.error(f"Lỗi gọi Gemini API: {e}")
        print(f"DEBUG ERROR: {e}")