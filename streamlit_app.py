# streamlit_app.py

import os
import json
import re
from textwrap import dedent

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# ———— 必须将 set_page_config 放在所有其他 Streamlit 命令之前 ————
st.set_page_config(
    page_title="SaleSeer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ———— Optional: LLM Support ————
OPENAI_KEY_AVAILABLE = True
openai_client = None

try:
    # Try to import and setup OpenAI with OpenRouter
    import openai  # type: ignore

    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("⚠️ OPENROUTER_API_KEY environment variable not set. Please set it to use AI features.")
        OPENAI_KEY_AVAILABLE = False
        openai_client = None
    else:
        # Configure OpenRouter API
        openai.api_key = api_key
        openai.api_base = "https://openrouter.ai/api/v1"

        if hasattr(openai, 'OpenAI'):
            # New OpenAI client
            openai_client = openai.OpenAI(api_key=openai.api_key, base_url=openai.api_base)  # type: ignore
        else:
            # Legacy OpenAI
            openai_client = openai
except ImportError:
    # OpenAI not available, continue without it
    OPENAI_KEY_AVAILABLE = False
except Exception:
    # Any other OpenAI setup issues
    OPENAI_KEY_AVAILABLE = False

# ———— Local parsing functions & TF-IDF modules —————
from inv_parser.inventory_parser import load_sample_inventory, load_csv
from recommendation.engine import get_recommendations
from recommendation.gpt_engine import generate_recommendation

# ———— Custom CSS styles —————
st.markdown("""
<style>
    /* Main background and global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Main title styles */
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
    
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Divider with status text */
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
    
    /* Chat header */
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
    
    /* Last query display */
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
    
    /* Result cards */
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
    
    /* Input box styling */
    .stChatInput > div > div > div > div {
        background-color: white;
        border: 2px solid #e5e7eb;
        border-radius: 1rem;
    }
    
    .stChatInput > div > div > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Success/info message styles */
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
    
    /* Preview caption */
    .preview-caption {
        font-size: 0.8rem;
        color: #6b7280;
        text-align: center;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ———— Main title area —————
st.markdown('<h1 class="main-title">🛒&nbsp;&nbsp;SaleSeer</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">AI Product Recommendation Assistant – Smart Inventory Analysis & Personalized Recommendations</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"><span class="divider-text">Ready for your query 🚀</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("📥 **Upload your inventory CSV, or click Load sample inventory to try the Demo**")

# ———— 0. Sidebar: Inventory cache/loading —————
if "inventory" not in st.session_state:
    st.session_state.inventory = None

with st.sidebar:
    st.markdown("### 📊 Inventory Management")
    
    if st.session_state.inventory is not None:
        inventory_count = len(st.session_state.inventory)
        st.markdown(f'<div class="success-box">✅ Loaded {inventory_count} products (cached)</div>', unsafe_allow_html=True)
        
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
            inventory_count = len(inv)
            st.markdown(f'<div class="success-box">✅ Loaded {inventory_count} sample products</div>', unsafe_allow_html=True)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("**Or upload your CSV file:**")
        upload = st.file_uploader("Upload inventory CSV", type="csv", key="csv_upload")
        if upload:
            inv = load_csv(upload)
            st.session_state.inventory = inv
            inventory_count = len(inv)
            st.markdown(f'<div class="success-box">✅ Loaded {inventory_count} products</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Please upload an inventory CSV, or click "Load sample inventory" above to get started.</div>', unsafe_allow_html=True)
            st.stop()  # Stop when no inventory is available

    st.markdown('<div class="divider"><span class="divider-text">Inventory Data</span></div>', unsafe_allow_html=True)
    
    # Sidebar preview
    st.markdown("### 🔍 Inventory Preview")
    with st.expander("View data details", expanded=False):
        preview_df = st.session_state.inventory.head(10)
        st.dataframe(preview_df, use_container_width=True)
        total_count = len(st.session_state.inventory)
        st.markdown(f'<p class="preview-caption">Showing first 10 of {total_count} products</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"><span class="divider-text">Actions</span></div>', unsafe_allow_html=True)
    
    # Reset button
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("🔄 Reset Chat & Inventory", key="reset_all", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ———— 1. Initialize session state —————
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None


# ———— 2. User preference parsing (budget + keywords) —————
def parse_user_query(query: str) -> dict:
    """
    Parse user input into:
      - prefs['budget_min'] and prefs['budget_max']: for range queries like "110 to 120"
      - prefs['budget']: numeric budget (for single bound queries)
      - prefs['budget_type']: "max" or "min", corresponding to <= budget or >= budget
      - prefs['keywords']: keyword list (example)
    """
    prefs = {"budget": None, "budget_type": "max", "budget_min": None, "budget_max": None, "keywords": []}

    # 1) First check for range patterns like "110 to 120" or "110-120"
    range_match = re.search(r"\$?(\d{1,7})\s*(?:to|-|~)\s*\$?(\d{1,7})", query, re.IGNORECASE)
    if range_match:
        # Found a range pattern
        min_price = int(range_match.group(1))
        max_price = int(range_match.group(2))
        # Ensure min <= max
        if min_price <= max_price:
            prefs["budget_min"] = min_price
            prefs["budget_max"] = max_price
        else:
            # Swap if user entered them backwards
            prefs["budget_min"] = max_price
            prefs["budget_max"] = min_price
        # Don't set single budget for range queries
        return prefs

    # 2) If no range found, look for single budget number (existing logic)
    m = re.search(r"\$?(\d{1,7})", query)
    if m:
        prefs["budget"] = int(m.group(1))

    # 3) Determine semantic meaning "above/over/greater" vs "below/under/less"
    lower_q = query.lower()
    if re.search(r"\b(above|over|greater|>=)\b", lower_q):
        prefs["budget_type"] = "min"  # means price >= budget
    elif re.search(r"\b(below|under|less|<=)\b", lower_q):
        prefs["budget_type"] = "max"  # means price <= budget
    # If neither matches, default to "max" (<= budget)

    # 4) Simple keyword extraction example (can be extended)
    for kw in ["red", "blue", "green", "dress", "shoe", "tech", "bag"]:
        if kw in lower_q:
            prefs["keywords"].append(kw)

    return prefs


# ———— 3. Rule filtering: budget + keywords (updated for range support) —————
def rule_based_recommend(inv: pd.DataFrame, prefs: dict, top_k: int = 5) -> pd.DataFrame:
    """
    Filter based on budget range or single budget bound in prefs:
      - If budget_min and budget_max are set: budget_min <= price <= budget_max
      - Otherwise: budget_type == "max": price <= budget, budget_type == "min": price >= budget
    Then apply keyword filtering.
    """
    df = inv.copy()

    # Budget filtering - check for range first
    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None and "price" in df.columns:
        # Range filtering
        df = df[(df["price"] >= prefs["budget_min"]) & (df["price"] <= prefs["budget_max"])]
    elif prefs.get("budget") is not None and "price" in df.columns:
        # Single bound filtering (existing logic)
        if prefs["budget_type"] == "max":
            df = df[df["price"] <= prefs["budget"]]
        else:  # "min"
            df = df[df["price"] >= prefs["budget"]]

    # Keyword filtering (search in name/category/color columns)
    if prefs["keywords"]:
        pattern = "|".join(prefs["keywords"])
        cols = [c for c in df.columns if c in ("name", "category", "color")]
        if cols:
            df = df[
                df[cols]
                .astype(str)
                .apply(lambda row: row.str.contains(pattern, case=False))
                .any(axis=1)
            ]

    return df.head(top_k)  # Only take top_k rows


# ———— 4. Helper function to apply price range filter —————
def apply_price_filter(inv: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    """
    Apply price range filtering to inventory DataFrame
    """
    df = inv.copy()

    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None and "price" in df.columns:
        # Range filtering
        df = df[(df["price"] >= prefs["budget_min"]) & (df["price"] <= prefs["budget_max"])]
    elif prefs.get("budget") is not None and "price" in df.columns:
        # Single bound filtering
        if prefs["budget_type"] == "max":
            df = df[df["price"] <= prefs["budget"]]
        else:  # "min"
            df = df[df["price"] >= prefs["budget"]]

    return df


# ———— 5. LLM fallback with price range filtering —————
def llm_recommend(inv: pd.DataFrame, query: str, prefs: dict) -> str:
    """
    Convert filtered inventory to JSON and send to LLM for recommendations
    """
    if not OPENAI_KEY_AVAILABLE or openai_client is None:
        return "*OpenAI service unavailable, please check configuration or use other recommendation modes*"

    # Apply price range filter first
    filtered_inv = apply_price_filter(inv, prefs)

    if filtered_inv.empty:
        if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
            return f"*No products found in price range ${prefs['budget_min']} to ${prefs['budget_max']}*"
        else:
            return "*No products found matching price criteria*"

    sample = filtered_inv.head(20).to_dict(orient="records")

    # Enhanced prompt with price range information
    prompt_prefix = f"User query: {query}"
    if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
        prompt_prefix += f"\nPrice range: ${prefs['budget_min']} to ${prefs['budget_max']}"

    messages = [
        {
            "role": "system",
            "content": "You are a friendly and professional AI shopping advisor who generates short recommendations based on JSON inventory and user needs. Only recommend products from the provided inventory."
        },
        {
            "role": "user",
            "content": f"{prompt_prefix}\n\nInventory JSON: {json.dumps(sample, ensure_ascii=False)}"
        },
    ]

    try:
        # Try different OpenAI API methods
        if hasattr(openai_client, 'chat') and hasattr(openai_client.chat, 'completions'):  # type: ignore
            # New OpenAI client
            response = openai_client.chat.completions.create(  # type: ignore
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )
        else:
            # Legacy OpenAI
            response = openai_client.ChatCompletion.create(  # type: ignore
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

        return response.choices[0].message.content.strip()  # type: ignore
    except Exception as e:
        return f"*LLM call failed: {str(e)}*"


# ———— 6. Recommendation execution function —————
def execute_recommendation(query: str, mode: str, inventory: pd.DataFrame) -> None:
    """
    Execute recommendation based on the selected mode and display results
    """
    # Parse preferences
    prefs = parse_user_query(query)

    # Execute based on mode
    if mode == "Rule-based":
        filtered = rule_based_recommend(inventory, prefs, top_k=3)
        if not filtered.empty:
            lines = []
            for i, (idx, row) in enumerate(filtered.iterrows(), 1):
                reason = []
                # Mark reasons based on budget range or single bound
                if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
                    # Range query
                    if prefs["budget_min"] <= row["price"] <= prefs["budget_max"]:
                        reason.append(f"Fits budget ({prefs['budget_min']}–{prefs['budget_max']})")
                elif prefs.get("budget") is not None:
                    # Single bound query
                    if prefs["budget_type"] == "max" and row["price"] <= prefs["budget"]:
                        reason.append(f"Fits budget (≤{prefs['budget']})")
                    if prefs["budget_type"] == "min" and row["price"] >= prefs["budget"]:
                        reason.append(f"Fits budget (≥{prefs['budget']})")
                if prefs["keywords"]:
                    reason.append("Matches keywords")
                lines.append(
                    f"{i}. **{row['name']} – ${row['price']}**  \n"
                    f"   Reason: {', '.join(reason) or 'Top pick'}"
                )

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🔍 Rule-based Top picks</div>', unsafe_allow_html=True)
            reply_md = "\n\n".join(lines)
            st.markdown(reply_md)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No products found matching the rules, please try changing budget or keywords.")

    elif mode == "TF-IDF + GPT":
        # Pre-filter inventory by price range before TF-IDF
        filtered_inventory = apply_price_filter(inventory, prefs)

        if filtered_inventory.empty:
            if prefs.get("budget_min") is not None and prefs.get("budget_max") is not None:
                st.warning(f"⚠️ No products found in price range ${prefs['budget_min']} to ${prefs['budget_max']}. Try adjusting your budget range.")
            else:
                st.warning("⚠️ No products found matching price criteria.")
            return

        # Now run TF-IDF on the filtered inventory
        tfidf_df = get_recommendations(query, filtered_inventory, k=3)

        if not tfidf_df.empty:
            top_k_list = tfidf_df[["name", "price"]].to_dict(orient="records")
            gpt_reply, model_used = generate_recommendation(query, top_k_list)

            # Use column layout to display TF-IDF results and GPT recommendations separately
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-header">🔢 TF-IDF Top 3 Matches</div>', unsafe_allow_html=True)
                st.dataframe(
                    tfidf_df[["name", "price", "similarity", "description"]],
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                # Create header with fallback warning if needed
                if model_used == "gpt-3.5-turbo":
                    st.markdown('<div class="result-header">🤖 GPT Smart Recommendation ⚠️ Using GPT-3.5 due to quota limits</div>', unsafe_allow_html=True)
                    st.markdown('💡 [Check your usage ↗️](https://platform.openai.com/account/usage)', unsafe_allow_html=True)
                    st.info("Note: Automatically fell back to GPT-3.5 due to quota or rate limits on GPT-4o-mini.")
                elif model_used == "error":
                    st.markdown('<div class="result-header">🤖 GPT Smart Recommendation ❌ Error</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-header">🤖 GPT Smart Recommendation</div>', unsafe_allow_html=True)

                st.markdown(gpt_reply)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No TF-IDF matches found. The query didn't match any products with sufficient similarity. Try different keywords or check your spelling.")

    else:  # LLM fallback
        if OPENAI_KEY_AVAILABLE:
            reply_md = llm_recommend(inventory, query, prefs)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-header">🧠 LLM Universal Recommendation</div>', unsafe_allow_html=True)
            st.markdown(reply_md)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ OpenAI Key unavailable, cannot call large language model.")

# ———— 7. Main conversation logic & interface —————
st.markdown('<h2 class="chat-header">💬 Let\'s Chat!</h2>', unsafe_allow_html=True)

# 7.1: User input box
user_msg = st.chat_input("Tell me what you need (e.g. color, budget, category or any natural language)", key="chat_input")

# 7.2: Handle new user input
if user_msg:
    # Store the new query
    st.session_state.last_query = user_msg
    st.session_state.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

# 7.3: Display last query and mode selection (always show if there's a last query)
if st.session_state.last_query:
    # Display the last query
    st.markdown('<div class="last-query-container">', unsafe_allow_html=True)
    st.markdown('<p class="last-query-text">🔄 Current Query:</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="query-content">{st.session_state.last_query}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode selection container
    st.markdown('<div class="mode-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="mode-title">Please select recommendation mode</h3>', unsafe_allow_html=True)
    
    # Create columns for mode selection and rerun button
    mode_col, button_col = st.columns([3, 1])
    
    with mode_col:
        mode = st.radio(
            "Mode",
            options=["Rule-based", "TF-IDF + GPT", "LLM fallback"],
            index=0,  # Default to Rule-based
            horizontal=True,
            key="mode_selection"
        )
    
    with button_col:
        rerun_clicked = st.button("🔄 Rerun", key="rerun_btn", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 7.4: Execute recommendation (either from new input or rerun button)
    if user_msg or rerun_clicked:
        query_to_process = st.session_state.last_query
        
        # Add to chat if it's a rerun (new queries already added above)
        if rerun_clicked and not user_msg:
            st.session_state.chat.append(("user", f"🔄 Rerun: {query_to_process}"))
            st.chat_message("user").write(f"🔄 **Rerun:** {query_to_process}")
        
        # Execute the recommendation
        execute_recommendation(query_to_process, mode, st.session_state.inventory)
        
        # Add to session state
        st.session_state.chat.append(("assistant", f"Recommendation generated using {mode} mode"))

# Display chat history
if st.session_state.chat:
    st.markdown('<div class="divider"><span class="divider-text">Conversation History</span></div>', unsafe_allow_html=True)
    st.markdown("### 💭 Recent Interactions")
    for role, content in st.session_state.chat[-6:]:  # Show last 6 messages
        if role == "user":
            st.chat_message("user").write(f"**User:** {content}")
        else:
            st.chat_message("assistant").write(f"**Assistant:** {content}")