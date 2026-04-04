import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Data files live next to this script; Streamlit Cloud cwd is often the repo root,
# so relative paths like "campaigns.json" fail and crash before any UI → blank page.
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Ad Campaign Finder",
    page_icon="🎯",
    layout="wide")



def _campaign_text(c: dict) -> str:
    """
    Build searchable text từ new schema.
    Gộp tất cả fields quan trọng thành 1 string để embed + BM25 index.
    Bây giờ include cả ai_enrichment fields → search by concept, tactic, audience.
    """
    meta = c.get("metadata", {})
    content = c.get("content", {})
    ai = c.get("ai_enrichment", {})

    tactics = ai.get("execution_tactics", "")
    if isinstance(tactics, list):
        tactics = " ".join(tactics)

    return (
        f"{meta.get('title', '')} "
        f"{meta.get('brand', '')} "
        f"{meta.get('agency', '')} "
        f"{meta.get('country', '')} "
        f"{meta.get('industry', '')} "
        f"{meta.get('medium', '')} "
        f"{content.get('description', '')} "
        f"{ai.get('concept_summary', '')} "
        f"{ai.get('target_audience', '')} "
        f"{tactics} "
        f"{ai.get('objective', '')}"
    )


@st.cache_resource
def load_search_stack():
    """Load campaigns + index. Uses embeddings.npy only if present locally (not in git)."""
    data_path = BASE_DIR / "campaigns_v2.json"
    with open(data_path, "r", encoding="utf-8") as f:
        campaigns = json.load(f)
    
    # BM25: keyword search — tốt cho exact match (brand name, country)
    corpus = [_campaign_text(c).lower().split() for c in campaigns]
    bm25 = BM25Okapi(corpus)

    # Sentence transformer: semantic search — tốt cho concept, meaning
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_path = BASE_DIR / "embeddings_v2.npy"
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
        np.save(emb_path, embeddings)

    return campaigns, embeddings, bm25, model



def hybrid_search(query, campaigns, bm25, embeddings, model, top_k=5, bm25_weight=0.5, semantic_weight=0.5):
    bm25_scores = bm25.get_scores(query.lower().split())

    query_embedding = model.encode([query], normalize_embeddings=True)
    semantic_scores = np.dot(embeddings, query_embedding.T).flatten()

    def normalize(scores):
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-9)

    final_scores = 0.5 * normalize(bm25_scores) + 0.5 * normalize(semantic_scores)
    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = campaigns[idx]
        meta = c.get("metadata", {})
        content = c.get("content", {})
        ai = c.get("ai_enrichment", {})
        results.append(
            {
            "score":             round(float(final_scores[idx]), 4),
            "id":                meta.get("id"),
            "title":             meta.get("title", ""),
            "brand":             meta.get("brand", ""),
            "agency":            meta.get("agency", ""),
            "industry":          meta.get("industry", ""),
            "country":           meta.get("country", ""),
            "medium":            meta.get("medium", ""),
            "thumbnail_url":     content.get("thumbnail_url", ""),
            "description":       content.get("description", ""),
            "url":               meta.get("url", ""),
            "concept_summary":   ai.get("concept_summary", ""),
            "target_audience":   ai.get("target_audience", ""),
            "execution_tactics": ai.get("execution_tactics", ""),
            "objective":         ai.get("objective", ""),
            }
        )
    return results

def init_session_state():
    """
    Session state là bộ nhớ tạm của Streamlit trong một session.
    Mỗi khi user tương tác (click, type), Streamlit re-run toàn bộ script.
    Session state giữ lại data giữa các lần re-run đó.
    
    favourites: dict {id: campaign} — dùng dict để tránh duplicate
    search_results: list kết quả search hiện tại
    messages: lịch sử chat
    conversation_history: lịch sử gửi cho Gemini
    """
    if "favourites" not in st.session_state:
        st.session_state.favourites = {}
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def render_campaign_card(campaign, show_favourite_btn=True):
    """
    Render 1 campaign card với thumbnail, title, brand, short desc.
    Dùng st.expander để ẩn/hiện full info.
    Trái tim button toggle favourite.
    """
    cid = campaign["id"]
    is_fav = cid in st.session_state.favourites

    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])

        # Thumbnail bên trái
        with col_img:
            if campaign.get("thumbnail_url"):
                st.image(campaign["thumbnail_url"], use_container_width=True)
            else:
                st.markdown("🎬")

        # Info bên phải
        with col_info:
            # Title + heart button cùng hàng
            title_col, heart_col = st.columns([5, 1])
            with title_col:
                st.markdown(f"### {campaign['title']}")
                st.markdown(f"**{campaign['brand']}** · {campaign['industry']} · {campaign['country']}")
            with heart_col:
                if show_favourite_btn:
                    heart = "❤️" if is_fav else "🤍"
                    if st.button(heart, key=f"fav_{cid}"):
                        if is_fav:
                            del st.session_state.favourites[cid]
                        else:
                            st.session_state.favourites[cid] = campaign
                        st.rerun()  # refresh UI ngay để update heart icon

            # Short description
            desc = campaign.get("description", "")
            st.markdown(desc[:150] + "..." if len(desc) > 150 else desc)

            # Expander cho full info
            with st.expander("See more"):
                st.markdown(f"**Agency:** {campaign.get('agency', 'N/A')}")
                st.markdown(f"**Medium:** {campaign.get('medium', 'N/A')}")
                st.markdown(f"**Concept:** {campaign.get('concept_summary', 'N/A')}")
                st.markdown(f"**Target Audience:** {campaign.get('target_audience', 'N/A')}")
                st.markdown(f"**Objective:** {campaign.get('objective', 'N/A')}")
                tactics = campaign.get("execution_tactics", "")
                if isinstance(tactics, list):
                    for t in tactics:
                        st.markdown(f"- {t}")
                else:
                    st.markdown(f"**Tactics:** {tactics}")
                st.markdown(f"[View Campaign ↗]({campaign.get('url', '')})")

def format_campaigns_as_context(results):
    context = ""
    for i, r in enumerate(results, 1):
        context += f"""
Campaign {i}:
- Title: {r['title']}
- Brand: {r['brand']}
- Industry: {r['industry']}
- Concept: {r['concept_summary']}
- Tactics: {r['execution_tactics']}
- Description: {r['description']}
- URL: {r['url']}
---
"""
    return context

def render_chatbot(campaigns, bm25, embeddings, model, google_client):
    """
    Chatbot panel bên phải.
    Giữ nguyên logic Gemini API của bạn — chỉ thêm system prompt tốt hơn.
    """
    st.markdown("### 🤖 Campaign Assistant")
    st.caption("Ask me to find campaigns, analyze briefs, or compare strategies")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("e.g. Find me funny food campaigns in Asia...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Search relevant campaigns to give LLM context
        results = hybrid_search(user_input, campaigns, bm25, embeddings, model, top_k=8)
        context = format_campaigns_as_context(results)

        system_prompt = f"""You are an expert marketing campaign analyst assistant.
You have access to a database of {len(campaigns)} real advertising campaigns.
Help users find relevant campaigns, analyze creative strategies, and match briefs to references.

When answering:
- Always mention campaign title, brand, and URL
- Explain WHY a campaign is relevant to the user's query
- If user pastes a brief, find the most relevant campaigns and explain the match
- Be concise but insightful

Retrieved campaigns for this query:
{context}"""

        user_message = {"role": "user", "parts": [{"text": user_input}]}
        current_contents = st.session_state.conversation_history + [user_message]

        # Giữ nguyên API call của bạn
        try:
            response = google_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=current_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1000,
                ),
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
            st.error(f"API Error: {e}")


def main():
    init_session_state()

    # Load data
    try:
        with st.spinner("Loading..."):
            campaigns, embeddings, bm25, model = load_search_stack()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # API client — giữ nguyên logic của bạn
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY")
        st.stop()
    google_client = genai.Client(api_key=api_key)

    # Tabs
    tab_search, tab_favourites = st.tabs(["🔍 Discover", "❤️ Favourites"])

    with tab_search:
        # Split layout: results left, chatbot right
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown("## Find Campaigns")
            query = st.text_input(
                "Search",
                placeholder="e.g. emotional storytelling for food brands in Southeast Asia",
                label_visibility="collapsed"
            )

            if query:
                results = hybrid_search(query, campaigns, bm25, embeddings, model)
                st.session_state.search_results = results
                st.caption(f"Found {len(results)} relevant campaigns")
                for r in results:
                    render_campaign_card(r)
            else:
                st.caption("Search to discover campaigns from our database of 424 enriched campaigns.")

        with right_col:
            render_chatbot(campaigns, bm25, embeddings, model, google_client)

    with tab_favourites:
        st.markdown("## ❤️ Saved Campaigns")
        if not st.session_state.favourites:
            st.info("No favourites yet. Click 🤍 on any campaign to save it here.")
        else:
            st.caption(f"{len(st.session_state.favourites)} campaigns saved")
            for cid, campaign in st.session_state.favourites.items():
                render_campaign_card(campaign, show_favourite_btn=True)

if __name__ == "__main__":
    main()


"""
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
"""