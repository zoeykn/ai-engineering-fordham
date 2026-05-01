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

def extract_filter_options(campaigns):
    """
    Extract sorted unique values for each filter field.
    Empty strings are excluded. Year is extracted from 'Month, YYYY' format.
    """
    years = set()
    countries = set()
    industries = set()
    mediums = set()
 
    for c in campaigns:
        meta = c.get("metadata", {})
 
        # Extract year from "January, 2026" format
        pub_date = meta.get("published_date", "")
        if pub_date:
            parts = pub_date.split(", ")
            if len(parts) == 2:
                years.add(parts[1])
 
        country = meta.get("country", "")
        if country:
            countries.add(country)
 
        industry = meta.get("industry", "")
        if industry:
            industries.add(industry)
 
        medium = meta.get("medium", "")
        if medium:
            mediums.add(medium)
 
    return {
        "years": sorted(years, reverse=True),       # newest first
        "countries": sorted(countries),
        "industries": sorted(industries),
        "mediums": sorted(mediums),
    }

def flatten_campaign(c: dict) -> dict:
    """
    Convert a raw JSON campaign object into a flat dict
    matching the format used by render_campaign_card().
    """
    meta = c.get("metadata", {})
    content = c.get("content", {})
    ai = c.get("ai_enrichment", {})
    return {
        "id":                meta.get("id"),
        "title":             meta.get("title", ""),
        "brand":             meta.get("brand", ""),
        "agency":            meta.get("agency", ""),
        "industry":          meta.get("industry", ""),
        "country":           meta.get("country", ""),
        "medium":            meta.get("medium", ""),
        "published_date":    meta.get("published_date", ""),
        "thumbnail_url":     content.get("thumbnail_url", ""),
        "description":       content.get("description", ""),
        "url":               meta.get("url", ""),
        "concept_summary":   ai.get("concept_summary", ""),
        "target_audience":   ai.get("target_audience", ""),
        "execution_tactics": ai.get("execution_tactics", ""),
        "objective":         ai.get("objective", ""),
    }
 
def apply_filters(campaigns_flat, selected_years, selected_countries, selected_industries, selected_mediums):
    """
    Filter a list of flattened campaigns by the selected filter values.
    If a filter list is empty, it means 'show all' (no restriction).
    """
    filtered = campaigns_flat
 
    if selected_years:
        filtered = [
            c for c in filtered
            if any(c.get("published_date", "").endswith(y) for y in selected_years)
        ]
    if selected_countries:
        filtered = [c for c in filtered if c.get("country", "") in selected_countries]
    if selected_industries:
        filtered = [c for c in filtered if c.get("industry", "") in selected_industries]
    if selected_mediums:
        filtered = [c for c in filtered if c.get("medium", "") in selected_mediums]
 
    return filtered
 
 

def init_session_state():
    """
    Session state là bộ nhớ tạm của Streamlit trong một session.
    Mỗi khi user tương tác (click, type), Streamlit re-run toàn bộ script.
    Session state giữ lại data giữa các lần re-run đó.
    
    favourites: dict {id: campaign} — use dict to avoid duplicate
    search_results: list search results
    messages: chat history
    conversation_history: history sent to Gemini
    """
    if "favourites" not in st.session_state:
        st.session_state.favourites = {}
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def render_campaign_card(campaign, show_favourite_btn=True, context="search"):
    """
    Render 1 campaign card with thumbnail, title, brand, short desc.
    Use st.expander to show/hide full info.
    Heart button toggle favourite.
    """
    cid = campaign["id"]
    is_fav = cid in st.session_state.favourites

    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])

        # Thumbnail on the left
        with col_img:
            if campaign.get("thumbnail_url"):
                st.image(campaign["thumbnail_url"], use_container_width=True)
            else:
                st.markdown("🎬")

        # Info on the right
        with col_info:
            # Title + heart button in the same row
            title_col, heart_col = st.columns([5, 1])
            with title_col:
                st.markdown(f"### {campaign['title']}")
                st.markdown(f"**{campaign['brand']}** · {campaign['industry']} · {campaign['country']}")
            with heart_col:
                if show_favourite_btn:
                    heart = "❤️" if is_fav else "🤍"
                    if st.button(heart,  key=f"fav_{context}_{cid}"):
                        if is_fav:
                            del st.session_state.favourites[cid]
                        else:
                            st.session_state.favourites[cid] = campaign
                        st.rerun()  # refresh UI immediately to update heart icon

            # Short description
            desc = campaign.get("description", "")
            st.markdown(desc[:150] + "..." if len(desc) > 150 else desc)

            # Expander for full info
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
        context += f"Campaign {i}: {r['title']} by {r['brand']} ({r['country']} — {r['concept_summary']} — URL: {r['url']}\n"
    return context

#tools for chatbot
def analyze_campaigns_data(campaigns, query_type, group_by=None, filter_field=None, filter_value=None, top_n=10):
    """
    Run aggregate queries on the campaign dataset.
    
    query_type: "count", "group_count", "list_unique", "top_n"
    group_by: field to group by (e.g. "country", "industry", "medium")
    filter_field/filter_value: optional filter before aggregation
    top_n: number of results for top_n queries
    """
    from collections import Counter

    # Flatten all campaigns
    flat = []
    for c in campaigns:
        meta = c.get("metadata", {})
        ai = c.get("ai_enrichment", {})
        flat.append({
            "title": meta.get("title", ""),
            "brand": meta.get("brand", ""),
            "agency": meta.get("agency", ""),
            "country": meta.get("country", ""),
            "industry": meta.get("industry", ""),
            "medium": meta.get("medium", ""),
            "published_date": meta.get("published_date", ""),
            "execution_tactics": ai.get("execution_tactics", ""),
            "objective": ai.get("objective", ""),
        })

    # Apply optional filter
    if filter_field and filter_value:
        flat = [c for c in flat if filter_value.lower() in c.get(filter_field, "").lower()]

    if query_type == "count":
        return {"total": len(flat)}

    elif query_type == "group_count":
        if not group_by:
            return {"error": "group_by field required"}
        counts = Counter(c.get(group_by, "Unknown") for c in flat)
        sorted_counts = dict(counts.most_common(top_n))
        return {"group_by": group_by, "results": sorted_counts, "total_groups": len(counts)}

    elif query_type == "list_unique":
        if not group_by:
            return {"error": "group_by field required"}
        unique = sorted(set(c.get(group_by, "") for c in flat if c.get(group_by)))
        return {"field": group_by, "unique_values": unique, "count": len(unique)}

    elif query_type == "top_n":
        if not group_by:
            return {"error": "group_by field required"}
        counts = Counter(c.get(group_by, "Unknown") for c in flat)
        return {"top": dict(counts.most_common(top_n))}

    return {"error": f"Unknown query_type: {query_type}"}

def render_chatbot(campaigns, bm25, embeddings, model, google_client):
    """
    Chatbot with Gemini + function calling.
    Tool: analyze_data — runs aggregate queries on campaign dataset.
    """
    st.markdown("### 🤖 Campaign Assistant")
    st.caption("Ask me to find campaigns, analyze briefs, or compare strategies")

    # Auto greeting
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi, I'm BraTo - Your Ad Buddy. How should we start today?"
        })

    # Scrollable chat history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("e.g. How many Film campaigns from Thailand?")

    if not user_input:
        return

    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Search relevant campaigns for context
    results = hybrid_search(user_input, campaigns, bm25, embeddings, model, top_k=5)
    context = format_campaigns_as_context(results)

    # Define the tool for Gemini
    analyze_tool_declaration = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="analyze_data",
            description="Run aggregate queries on the campaign database. Use this when user asks about counts, distributions, trends, comparisons across the dataset. Fields available: country, industry, medium, brand, agency, published_date.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query_type": types.Schema(
                        type="STRING",
                        description="Type of analysis: 'count' (total campaigns), 'group_count' (count by field), 'list_unique' (unique values of a field), 'top_n' (most common values)",
                        enum=["count", "group_count", "list_unique", "top_n"],
                    ),
                    "group_by": types.Schema(
                        type="STRING",
                        description="Field to group/aggregate by",
                        enum=["country", "industry", "medium", "brand", "agency"],
                    ),
                    "filter_field": types.Schema(
                        type="STRING",
                        description="Optional: field to filter on before aggregation",
                        enum=["country", "industry", "medium", "brand", "agency"],
                    ),
                    "filter_value": types.Schema(
                        type="STRING",
                        description="Optional: value to filter for",
                    ),
                    "top_n": types.Schema(
                        type="INTEGER",
                        description="Number of top results to return (default 10)",
                    ),
                },
                required=["query_type"],
            ),
        ),
    ])

    system_prompt = f"""You are BraTo, an expert marketing campaign analyst assistant.
You have access to a database of {len(campaigns)} real advertising campaigns.

You have two capabilities:
1. SEARCH: Retrieved campaigns are provided below for finding specific campaigns.
2. ANALYZE: Use the analyze_data tool when users ask about counts, trends, distributions, 
   or aggregate insights (e.g. "how many", "which country has most", "breakdown by medium").

When answering:
- Use analyze_data for any quantitative/aggregate question
- Use retrieved campaigns for specific campaign recommendations
- Always be concise but insightful
- Mention campaign titles, brands, and URLs when recommending

Retrieved campaigns for this query:
{context}"""

    user_message = {"role": "user", "parts": [{"text": user_input}]}
    current_contents = st.session_state.conversation_history + [user_message]

    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=current_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2500,
                tools=[analyze_tool_declaration, types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        # Handle function calling loop
        while response.candidates[0].content.parts:
            # Check if any part is a function call
            func_call = None
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    func_call = part
                    break

            if not func_call:
                break  # No function call, just text response

            # Execute the function
            fc = func_call.function_call
            if fc.name == "analyze_data":
                args = dict(fc.args)
                tool_result = analyze_campaigns_data(
                    campaigns,
                    query_type=args.get("query_type", "count"),
                    group_by=args.get("group_by"),
                    filter_field=args.get("filter_field"),
                    filter_value=args.get("filter_value"),
                    top_n=args.get("top_n", 10),
                )
            else:
                tool_result = {"error": f"Unknown function: {fc.name}"}

            # Send function result back to Gemini
            current_contents.append(response.candidates[0].content)
            current_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=fc.name,
                        response=tool_result,
                    ))],
                )
            )

            response = google_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=current_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2500,
                    tools=[analyze_tool_declaration, types.Tool(google_search=types.GoogleSearch())],
                ),
            )

        # Extract final text response
        assistant_message = response.text

        st.session_state.conversation_history.append(user_message)
        st.session_state.conversation_history.append(
            {"role": "model", "parts": [{"text": assistant_message}]}
        )

        with chat_container:
            with st.chat_message("assistant"):
                st.write(assistant_message)
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    except Exception as e:
        st.error(f"API Error: {e}")



#Pagination

CAMPAIGNS_PER_PAGE = 20

def render_paginated_campaigns(display_list, context="search"):
    """
    Render campaigns with pagination instead of endless scroll.
    Shows CAMPAIGNS_PER_PAGE items per page with Previous/Next buttons.
    """
    # Initialize page number in session state
    page_key = f"page_{context}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total = len(display_list)
    total_pages = max(1, (total + CAMPAIGNS_PER_PAGE - 1) // CAMPAIGNS_PER_PAGE)

    # Clamp current page if filters reduced total
    if st.session_state[page_key] >= total_pages:
        st.session_state[page_key] = 0

    page = st.session_state[page_key]
    start = page * CAMPAIGNS_PER_PAGE
    end = min(start + CAMPAIGNS_PER_PAGE, total)

    # Render cards for current page
    for c in display_list[start:end]:
        render_campaign_card(c, context=context)

    # Pagination controls
    if total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_info:
            st.caption(f"Page {page + 1} of {total_pages} · {total} campaigns")
        with col_prev:
            if st.button("← Previous", disabled=(page == 0), key=f"prev_{context}"):
                st.session_state[page_key] -= 1
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=(page >= total_pages - 1), key=f"next_{context}"):
                st.session_state[page_key] += 1
                st.rerun()

#Main function

def main():
    init_session_state()

    st.markdown("""
        <h1 style='font-size: 48px; font-weight: 1000; margin-bottom: 0;'>AdBuddy</h1>
    """, unsafe_allow_html=True)

    # Load data
    try:
        with st.spinner("Loading..."):
            campaigns, embeddings, bm25, model = load_search_stack()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Gemini API client
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY")
        st.stop()
    google_client = genai.Client(api_key=api_key)

    # Extract unique filter options once
    filter_opts = extract_filter_options(campaigns)

    # Flatten all campaigns for display and filtering
    all_campaigns_flat = [flatten_campaign(c) for c in campaigns]

    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab"] {
            font-size: 26px
            font-weight: 70
            padding: 10px 24px;
        }
        h2 { font-size: 18px !important; }
        </style>
    """, unsafe_allow_html=True)

    # Tabs
    tab_search, tab_favourites = st.tabs(["Discover", "Favourites"])

    with tab_search:
        # Main layout: left content (3) + right chatbot (1)
        col_main, col_chat = st.columns([3, 1])

        with col_main:
            st.markdown("#### Find Campaigns")

            # Search bar
            query = st.text_input(
                "Search",
                placeholder="e.g. emotional storytelling for food brands in Southeast Asia",
                label_visibility="collapsed",
            )

            # Filters: Year - Country - Industry - Medium
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                sel_years = st.multiselect("Year", filter_opts["years"])
            with f2:
                sel_countries = st.multiselect("Country", filter_opts["countries"])
            with f3:
                sel_industries = st.multiselect("Industry", filter_opts["industries"])
            with f4:
                sel_mediums = st.multiselect("Medium", filter_opts["mediums"])

            # Determine which campaigns to show
            if query:
                # Hybrid search first, then apply filters
                search_results = hybrid_search(query, campaigns, bm25, embeddings, model, top_k=50)
                display_list = apply_filters(search_results, sel_years, sel_countries, sel_industries, sel_mediums)
            else:
                # No search query — show all campaigns, apply filters only
                display_list = apply_filters(all_campaigns_flat, sel_years, sel_countries, sel_industries, sel_mediums)

            # Results count
            st.caption(f"Showing {len(display_list)} of {len(campaigns)} campaigns")

            # Render campaign cards
            render_paginated_campaigns(display_list, context="search")

        with col_chat:
            render_chatbot(campaigns, bm25, embeddings, model, google_client)

    with tab_favourites:
        st.markdown("## ❤️ Saved Campaigns")
        if not st.session_state.favourites:
            st.info("No favourites yet. Click 🤍 on any campaign to save it here.")
        else:
            st.caption(f"{len(st.session_state.favourites)} campaigns saved")
            for cid, campaign in st.session_state.favourites.items():
                render_campaign_card(campaign, show_favourite_btn=True, context="fav")

if __name__ == "__main__":
    main()

