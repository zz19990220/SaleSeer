# streamlit_app.py
import os
import json
import re

import streamlit as st
import pandas as pd

# —— Optional: LLM (OpenRouter) Support —— 
OPENAI_KEY_AVAILABLE = True  # Enable OpenAI functionality
openai_client = None

try:
    import openai  # type: ignore

    # 1) You can either load from an environment variable or hard-code for testing:
    #    If you want to load from a `.env` file, uncomment the two lines below and store your key in `.env`.
    # from dotenv import load_dotenv
    # load_dotenv()
    # openai_key = os.getenv("OPENAI_API_KEY")

    # In this code snippet, we'll hard-code your new OpenRouter key:
    openai_key = "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"

    if openai_key:
        # Set up OpenAI legacy API configuration for v0.28.0 with OpenRouter
        openai.api_key = openai_key
        openai.api_base = "https://openrouter.ai/api/v1"
        # Set the api_type to None for OpenRouter compatibility
        openai.api_type = None
        openai.api_version = None
        openai_client = openai  # Use legacy client for v0.28.0
        OPENAI_KEY_AVAILABLE = True
    else:
        OPENAI_KEY_AVAILABLE = False

except ImportError:
    OPENAI_KEY_AVAILABLE = False
except Exception:
    OPENAI_KEY_AVAILABLE = False

# —— Local parsing & TF-IDF modules —— 
from inv_parser.inventory_parser import load_sample_inventory, load_csv
from recommendation.engine import get_recommendations
from recommendation.gpt_engine import generate_recommendation

# —— Page configuration —— 
st.set_page_config(
    page_title="SaleSeer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# —— Custom CSS —— 
st.markdown("""
<style>
    /* Main container padding + background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f8f9fa;
    }
    /* Title styling */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    .main-subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Gradient divider styling */
    .divider {
        height: 3rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        margin: 1.5rem 0;
        border-radius: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .divider-text {
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    /* Button container */
    .button-container {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        margin: 1rem 0;
    }
    /* Chat header styling */
    .chat-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
    }
    /* Mode selection container */
    .mode-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .mode-title {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    /* Last query container */
    .last-query-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .last-query-text {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .query-content {
        font-size: 1rem;
        color: #1e293b;
        font-weight: 500;
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #cbd5e1;
    }
    /* Result card styling */
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .result-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    /* Chat input box styling */
    .stChatInput > div > div > div > div {
        background-color: white;
        border: 2px solid #e5e7eb;
        border-radius: 1rem;
    }
    .stChatInput > div > div > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    /* Success / Info message boxes */
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #93c5fd;
        color: #1e40af;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Preview caption for inventory table */
    .preview-caption {
        font-size: 0.8rem;
        color: #6b7280;
        text-align: center;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# —— Main title area —— 
st.markdown('<h1 class="main-title">🛒&nbsp;&nbsp;SaleSeer</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">AI Product Recommendation Assistant – Smart Inventory Analysis & Personalized Suggestions</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"><span class="divider-text">Ready for your query 🚀</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("📥 **Upload your inventory CSV, or click Load sample inventory to try the demo**")

# —— 0) Sidebar — Inventory Loading & Preview —— 
if "inventory" not in st.session_state:
    st.session_state.inventory = None

with st.sidebar:
    st.markdown("### 📊 Inventory Management")

    if st.session_state.inventory is not None:
        count = len(st.session_state.inventory)
        st.markdown(f'<div class="success-box">✅ Loaded {count} products (cached)</div>', unsafe_allow_html=True)
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("🔄 Reload Sample Inventory", key="reload_btn", use_container_width=True):
            inv = load_sample_inventory()
            st.session_state.inventory = inv
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("📦 Load sample inventory", key="load_sample", use_container_width=True):
            inv = load_sample_inventory()
            st.session_state.inventory = inv
            count = len(inv)
            st.markdown(f'<div class="success-box">✅ Loaded {count} sample products</div>', unsafe_allow_html=True)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("**Or upload your own inventory CSV:**")
        upload = st.file_uploader("Upload inventory CSV", type="csv", key="csv_upload")
        if upload:
            inv = load_csv(upload)
            st.session_state.inventory = inv
            count = len(inv)
            st.markdown(f'<div class="success-box">✅ Loaded {count} products</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Please upload an inventory CSV, or click "Load sample inventory" above to get started.</div>', unsafe_allow_html=True)
            st.stop()

    st.markdown('<div class="divider"><span class="divider-text">Inventory Data</span></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Inventory Preview")
    with st.expander("View first 10 rows", expanded=False):
        preview_df = st.session_state.inventory.head(10)
        st.dataframe(preview_df, use_container_width=True)
        total = len(st.session_state.inventory)
        st.markdown(f'<p class="preview-caption">Showing first 10 of {total} products</p>', unsafe_allow_html=True)

    st.markdown('<div class="divider"><span class="divider-text">Actions</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("🔄 Reset Chat & Inventory", key="reset_all", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# —— 1) Initialize session state for chat & last_query —— 
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None


# —— 2) Parse user query into structured preferences —— 
def parse_user_query(query: str, inventory: pd.DataFrame) -> dict:
    """
    Parse the user's free-text query into:
      - budget (single bound) or budget_min & budget_max (range)
      - category   (matched dynamically from inventory["category"])
      - color      (matched dynamically from inventory["color"])
      - keywords   (from a predefined list)
    """
    prefs = {
        "budget": None,
        "budget_type": "max",
        "budget_min": None,
        "budget_max": None,
        "category": None,
        "color": None,
        "keywords": []
    }

    lower_q = query.lower()

    # 1) Check for a numeric range "110 to 120" or "110-120"
    range_match = re.search(r"\$?(\d{1,7})\s*(?:to|-|~)\s*\$?(\d{1,7})", lower_q)
    if range_match:
        a = int(range_match.group(1))
        b = int(range_match.group(2))
        if a <= b:
            prefs["budget_min"], prefs["budget_max"] = a, b
        else:
            prefs["budget_min"], prefs["budget_max"] = b, a
        return prefs  # no single budget needed

    # 2) Single bound budget, e.g. ">$100" or "under 200"
    m = re.search(r"\$?(\d{1,7})", lower_q)
    if m:
        prefs["budget"] = int(m.group(1))
    if re.search(r"\b(above|over|greater|>=)\b", lower_q):
        prefs["budget_type"] = "min"
    elif re.search(r"\b(below|under|less|<=)\b", lower_q):
        prefs["budget_type"] = "max"

    # 3) Dynamic category matching from inventory["category"] (case-insensitive)
    unique_cats = inventory["category"].dropna().astype(str).unique().tolist()
    for cat in unique_cats:
        if cat.lower() in lower_q:
            prefs["category"] = cat
            break

    # 4) Dynamic color matching from inventory["color"]
    if "color" in inventory.columns:
        unique_colors = inventory["color"].dropna().astype(str).unique().tolist()
        for col in unique_colors:
            if col.lower() in lower_q:
                prefs["color"] = col
                break

    # 5) Predefined keyword list
    predefined_kw = [
        "red", "blue", "green", "dress", "shoe", "tech", "bag",
        "phone", "jeans", "sneakers", "headphones", "watch", "keyboard",
        "charger", "camera", "laptop", "fitbit", "wireless", "gaming", "smart", "budget"
    ]
    for kw in predefined_kw:
        if kw in lower_q:
            prefs["keywords"].append(kw)

    return prefs


# —— 3) Rule-based filtering (multi-step) —— 
def rule_based_recommend(inv: pd.DataFrame, prefs: dict, top_k: int = 5) -> pd.DataFrame:
    """
    Apply filters in this order:
      1) Price (range or single bound)
      2) Category (if specified)
      3) Color (if specified)
      4) Keywords (search in name/category/color columns)
    Return top_k results (head of DataFrame) or empty DataFrame if nothing matches.
    """
    df = inv.copy()

    # 1) Price filter
    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
        df = df[(df["price"] >= prefs["budget_min"]) & (df["price"] <= prefs["budget_max"])]
    elif prefs.get("budget") is not None:
        if prefs["budget_type"] == "max":
            df = df[df["price"] <= prefs["budget"]]
        else:
            df = df[df["price"] >= prefs["budget"]]

    # 2) Category filter
    if prefs.get("category"):
        df = df[df["category"].str.lower() == prefs["category"].lower()]

    # 3) Color filter
    if prefs.get("color") and "color" in df.columns:
        df = df[df["color"].str.lower() == prefs["color"].lower()]

    # 4) Keyword filter
    if prefs["keywords"] and not df.empty:
        pattern = "|".join(prefs["keywords"])
        cols = [c for c in df.columns if c in ("name", "category", "color")]
        if cols:
            df = df[
                df[cols]
                .astype(str)
                .apply(lambda row: row.str.contains(pattern, case=False))
                .any(axis=1)
            ]

    return df.head(top_k)


# —— 4) Apply price-only filter (for TF-IDF & LLM fallback) —— 
def apply_price_filter(inv: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    df = inv.copy()
    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
        df = df[(df["price"] >= prefs["budget_min"]) & (df["price"] <= prefs["budget_max"])]
    elif prefs.get("budget") is not None:
        if prefs["budget_type"] == "max":
            df = df[df["price"] <= prefs["budget"]]
        else:
            df = df[df["price"] >= prefs["budget"]]
    return df


# —— 5) LLM fallback logic —— 
def llm_recommend(inv: pd.DataFrame, query: str, prefs: dict) -> str:
    """
    1) Price filter
    2) Category filter
    3) Color filter
    4) Keyword filter
    5) If still non-empty, send top-20 JSON to OpenRouter chat.completions
    6) Return the assistant's reply or an error message
    """
    if not OPENAI_KEY_AVAILABLE or openai_client is None:
        return "*OpenAI service unavailable. Please check your configuration.*"

    # 1) Price
    filtered = apply_price_filter(inv, prefs)

    # 2) Category
    if prefs.get("category"):
        filtered = filtered[filtered["category"].str.lower() == prefs["category"].lower()]

    # 3) Color
    if prefs.get("color") and "color" in filtered.columns:
        filtered = filtered[filtered["color"].str.lower() == prefs["color"].lower()]

    # 4) Keywords
    if prefs["keywords"] and not filtered.empty:
        pattern = "|".join(prefs["keywords"])
        cols = [c for c in filtered.columns if c in ("name", "category", "color")]
        if cols:
            filtered = filtered[
                filtered[cols]
                .astype(str)
                .apply(lambda row: row.str.contains(pattern, case=False))
                .any(axis=1)
            ]

    # If nothing remains, return a friendly "no results" message
    if filtered.empty:
        if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
            return f"*No products found in price range ${prefs['budget_min']} to ${prefs['budget_max']}.*"
        if prefs.get("category"):
            return f"*No products found in category '{prefs['category']}'.*"
        if prefs.get("color"):
            return f"*No products found in color '{prefs['color']}'.*"
        return "*No products found matching the given criteria.*"

    # Take top‐20 rows for LLM prompt
    snippet = filtered.head(20).to_dict(orient="records")

    prompt_prefix = f"User query: {query}"
    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
        prompt_prefix += f"\nPrice range: ${prefs['budget_min']} to ${prefs['budget_max']}"
    if prefs.get("category"):
        prompt_prefix += f"\nCategory: {prefs['category']}"
    if prefs.get("color"):
        prompt_prefix += f"\nColor: {prefs['color']}"

    messages = [
        {
            "role": "system",
            "content": "You are a friendly, professional AI shopping advisor. Only recommend products from the provided inventory JSON."
        },
        {
            "role": "user",
            "content": f"{prompt_prefix}\n\nInventory JSON: {json.dumps(snippet, ensure_ascii=False)}"
        },
    ]

    try:
        # Import the requests-based wrapper for reliable OpenRouter calls
        from openai_wrapper import call_openrouter_chat
        
        # Use the custom wrapper instead of the OpenAI client
        response_text, model_used = call_openrouter_chat(messages, model="gpt-4o-mini", temperature=0.7)
        return response_text
    except Exception as e:
        # Return a clear error message to the user
        return f"*LLM call failed: {str(e)}. The OpenAI key may be invalid or expired. Please check your configuration.*"


# —— 6) "TF-IDF + GPT" and "Rule-based" execution —— 
def execute_recommendation(query: str, mode: str, inventory: pd.DataFrame) -> None:
    """
    1) Parse user query into structured prefs
    2) If Rule-based: run multi-step filtering (price/category/color/keywords)
       - If empty: show fallback Top-3 from entire inventory
       - Otherwise: show up to 3 matches with "reason" bullets
    3) If TF-IDF+GPT: run the same multi-step filtering (price/category/color/keywords) first,
       then run get_recommendations(...) on whatever remains,
       then (a) display a small TF-IDF table (name, price, similarity, description) and
            (b) call generate_recommendation(...) to get a GPT‐written sentence block, catching errors.
    4) If LLM fallback: call llm_recommend(...) as above.
    """
    prefs = parse_user_query(query, inventory)

    if mode == "Rule-based":
        filtered = rule_based_recommend(inventory, prefs, top_k=3)
        if not filtered.empty:
            lines = []
            for i, (idx, row) in enumerate(filtered.iterrows(), 1):
                reasons = []
                if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
                    if prefs["budget_min"] <= row["price"] <= prefs["budget_max"]:
                        reasons.append(f"Budget ${prefs['budget_min']}–${prefs['budget_max']}")
                elif prefs.get("budget") is not None:
                    if prefs["budget_type"] == "max" and row["price"] <= prefs["budget"]:
                        reasons.append(f"Budget ≤${prefs['budget']}")
                    if prefs["budget_type"] == "min" and row["price"] >= prefs["budget"]:
                        reasons.append(f"Budget ≥${prefs['budget']}")
                if prefs.get("category"):
                    reasons.append(f"Category {prefs['category']}")
                if prefs.get("color"):
                    reasons.append(f"Color {prefs['color']}")
                if prefs["keywords"]:
                    reasons.append("Matches keyword")
                if not reasons:
                    reasons.append("Top pick")
                lines.append(
                    f"{i}. **{row['name']} – ${row['price']}**  \n"
                    f"   Reason: {', '.join(reasons)}"
                )
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🔍 Rule-based Top picks</div>', unsafe_allow_html=True)
            st.markdown("\n\n".join(lines))
            st.markdown('</div>', unsafe_allow_html=True)
            return
        else:
            st.warning("⚠️ No products found matching criteria. Showing fallback Top 3.")
            fallback = inventory.head(3)
            lines = []
            for i, (idx, row) in enumerate(fallback.iterrows(), 1):
                lines.append(f"{i}. **{row['name']} – ${row['price']}**  \n   Reason: Top pick (fallback)")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🔍 Fallback Top picks</div>', unsafe_allow_html=True)
            st.markdown("\n\n".join(lines))
            st.markdown('</div>', unsafe_allow_html=True)
            return

    elif mode == "TF-IDF + GPT":
        # (A) Apply the same multi-step filter as rule-based:
        temp = apply_price_filter(inventory, prefs)
        if prefs.get("category"):
            temp = temp[temp["category"].str.lower() == prefs["category"].lower()]
        if prefs.get("color") and "color" in temp.columns:
            temp = temp[temp["color"].str.lower() == prefs["color"].lower()]
        if prefs["keywords"] and not temp.empty:
            pattern = "|".join(prefs["keywords"])
            cols = [c for c in temp.columns if c in ("name", "category", "color")]
            if cols:
                temp = temp[
                    temp[cols]
                    .astype(str)
                    .apply(lambda row: row.str.contains(pattern, case=False))
                    .any(axis=1)
                ]

        if temp.empty:
            st.warning("⚠️ No products found after filters. Showing fallback Top 3.")
            fallback = inventory.head(3)
            lines = []
            for i, (idx, row) in enumerate(fallback.iterrows(), 1):
                lines.append(f"{i}. **{row['name']} – ${row['price']}**  \n   Reason: Top pick (fallback)")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🔍 Fallback Top picks</div>', unsafe_allow_html=True)
            st.markdown("\n\n".join(lines))
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # (B) Use TF-IDF on what's left
        try:
            tfidf_df = get_recommendations(query, temp, k=3)
            st.info(f"🔍 TF-IDF found {len(tfidf_df) if tfidf_df is not None else 0} matches for '{query}'")
            
            if tfidf_df is not None and not tfidf_df.empty:
                # Check if required columns exist
                required_cols = ["name", "price", "similarity"]
                if not all(col in tfidf_df.columns for col in required_cols):
                    st.error(f"Missing columns in TF-IDF results. Expected: {required_cols}, Got: {list(tfidf_df.columns)}")
                    return
                    
                top_k_list = tfidf_df[["name", "price"]].to_dict(orient="records")
                
                # Try to get GPT recommendation
                try:
                    gpt_reply, model_used = generate_recommendation(query, top_k_list)
                except Exception as gpt_error:
                    st.error(f"GPT recommendation failed: {str(gpt_error)}")
                    gpt_reply = "*GPT recommendation unavailable. Please check your OpenAI configuration.*"
                    model_used = "error"

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown('<div class="result-header">🔢 TF-IDF Top 3 Matches</div>', unsafe_allow_html=True)
                    
                    # Display available columns safely
                    display_cols = [col for col in ["name", "price", "similarity", "description"] if col in tfidf_df.columns]
                    st.dataframe(
                        tfidf_df[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    if model_used == "gpt-3.5-turbo":
                        st.markdown('<div class="result-header">🤖 GPT Smart Recommendation ⚠️ Using GPT-3.5</div>', unsafe_allow_html=True)
                        st.markdown('💡 [Check usage ↗️](https://platform.openai.com/account/usage)', unsafe_allow_html=True)
                        st.info("Note: fell back to GPT-3.5 due to quota or rate limits.")
                    elif model_used == "demo-gpt":
                        st.markdown('<div class="result-header">🎯 GPT Smart Recommendation (Demo Mode)</div>', unsafe_allow_html=True)
                        st.info("Demo mode: OpenAI API key is invalid/expired. Please update your API key for live recommendations.")
                    elif model_used == "error":
                        st.markdown('<div class="result-header">🤖 GPT Smart Recommendation ❌ Error</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-header">🤖 GPT Smart Recommendation</div>', unsafe_allow_html=True)
                    st.markdown(gpt_reply)
                    st.markdown('</div>', unsafe_allow_html=True)
                return
            else:
                st.warning(f"⚠️ TF-IDF did not find any matches for '{query}'. This can happen if your query doesn't semantically match product names/descriptions. Showing fallback Top 3.")
                fallback = inventory.head(3)
                lines = []
                for i, (idx, row) in enumerate(fallback.iterrows(), 1):
                    lines.append(f"{i}. **{row['name']} – ${row['price']}**  \n   Reason: Top pick (fallback)")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-header">🔍 Fallback Top picks</div>', unsafe_allow_html=True)
                st.markdown("\n\n".join(lines))
                st.markdown('</div>', unsafe_allow_html=True)
                return
        except Exception as tfidf_error:
            st.error(f"TF-IDF engine error: {str(tfidf_error)}")
            st.warning("⚠️ TF-IDF failed. Showing fallback Top 3.")
            fallback = inventory.head(3)
            lines = []
            for i, (idx, row) in enumerate(fallback.iterrows(), 1):
                lines.append(f"{i}. **{row['name']} – ${row['price']}**  \n   Reason: Top pick (fallback)")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🔍 Fallback Top picks</div>', unsafe_allow_html=True)
            st.markdown("\n\n".join(lines))
            st.markdown('</div>', unsafe_allow_html=True)
            return

    else:  # LLM fallback
        if OPENAI_KEY_AVAILABLE:
            try:
                st.info(f"🧠 Calling LLM for universal recommendation...")
                reply_md = llm_recommend(inventory, query, prefs)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-header">🧠 LLM Universal Recommendation</div>', unsafe_allow_html=True)
                st.markdown(reply_md)
                st.markdown('</div>', unsafe_allow_html=True)
                return
            except Exception as llm_error:
                st.error(f"LLM recommendation failed: {str(llm_error)}")
                st.warning("⚠️ LLM fallback failed. Showing basic Top 3.")
                fallback = inventory.head(3)
                lines = []
                for i, (idx, row) in enumerate(fallback.iterrows(), 1):
                    lines.append(f"{i}. **{row['name']} – ${row['price']}**  \n   Reason: Top pick (fallback)")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-header">🔍 Fallback Top picks</div>', unsafe_allow_html=True)
                st.markdown("\n\n".join(lines))
                st.markdown('</div>', unsafe_allow_html=True)
                return
        else:
            st.error("⚠️ OpenAI Key unavailable. Cannot call large language model.")
            return


# —— 7) Main chat interface & logic —— 
st.markdown('<h2 class="chat-header">💬 Let\'s Chat!</h2>', unsafe_allow_html=True)

user_msg = st.chat_input("Tell me what you need (e.g. color, budget, category, or any natural language)", key="chat_input")

if user_msg:
    st.session_state.last_query = user_msg
    st.session_state.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

if st.session_state.last_query:
    # Show the last query
    st.markdown('<div class="last-query-container">', unsafe_allow_html=True)
    st.markdown('<p class="last-query-text">🔄 Current Query:</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="query-content">{st.session_state.last_query}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Mode selection + Rerun button
    st.markdown('<div class="mode-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="mode-title">Please select recommendation mode</h3>', unsafe_allow_html=True)
    mode_col, button_col = st.columns([3, 1])
    with mode_col:
        mode = st.radio(
            "Mode",
            options=["Rule-based", "TF-IDF + GPT", "LLM fallback"],
            index=0,
            horizontal=True,
            key="mode_selection"
        )
    with button_col:
        rerun_clicked = st.button("🔄 Rerun", key="rerun_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # If either new message or rerun is clicked, run the recommendation
    if user_msg or rerun_clicked:
        query_to_process = st.session_state.last_query
        if rerun_clicked and not user_msg:
            st.session_state.chat.append(("user", f"🔄 Rerun: {query_to_process}"))
            st.chat_message("user").write(f"🔄 **Rerun:** {query_to_process}")

        execute_recommendation(query_to_process, mode, st.session_state.inventory)
        st.session_state.chat.append(("assistant", f"Recommendation generated using {mode} mode"))

# Show chat history (last 6 messages)
if st.session_state.chat:
    st.markdown('<div class="divider"><span class="divider-text">Conversation History</span></div>', unsafe_allow_html=True)
    st.markdown("### 💭 Recent Interactions")
    for role, content in st.session_state.chat[-6:]:
        if role == "user":
            st.chat_message("user").write(f"**User:** {content}")
        else:
            st.chat_message("assistant").write(f"**Assistant:** {content}")