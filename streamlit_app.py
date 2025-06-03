import os
import json
import re
from textwrap import dedent

import streamlit as st
import pandas as pd

# ---------- 可选：如需 LLM 推荐 ----------
try:
    import openai

    OPENAI_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_KEY_AVAILABLE:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_KEY_AVAILABLE = False
# ----------------------------------------

from inv_parser.inventory_parser import load_sample_inventory, load_csv
from recommendation.engine import get_recommendations  # NEW ✅

st.set_page_config(page_title="SaleSeer", page_icon="🛒")
st.title("🛒 SaleSeer – AI Product Recommender")
st.write("Upload **your** inventory CSV or click *Load sample inventory* to try the demo.")

# -----------------------------------------------------------------------------------
# 0. Sidebar – Inventory 载入 & 价格区间 slider（Issue #4）
# -----------------------------------------------------------------------------------
with st.sidebar:
    # 0-1 载入 / 上传 CSV
    if "inventory" not in st.session_state:
        if st.button("Load sample inventory"):
            st.session_state.inventory = load_sample_inventory()
        else:
            csv_file = st.file_uploader("Upload inventory CSV", type="csv")
            if csv_file:
                st.session_state.inventory = load_csv(csv_file)
            else:
                st.info("Please upload an inventory CSV or click Load sample inventory.")
                st.stop()

    inventory = st.session_state.inventory
    st.success(f"{len(inventory)} products ready (cached)")

    # 0-2 价格 slider
    if "price" in inventory.columns and not inventory["price"].isna().all():
        min_price, max_price = int(inventory["price"].min()), int(inventory["price"].max())
        st.sidebar.subheader("Price filter")
        price_min, price_max = st.slider(
            "Select price range ($)", min_price, max_price, (min_price, max_price)
        )
    else:
        # 若无价格列，slider 无效
        price_min, price_max = None, None

    # 0-3 预览 & reset
    with st.expander("Preview inventory"):
        st.dataframe(inventory.head())

    if st.button("Reset chat & inventory"):
        st.session_state.clear()
        st.experimental_rerun()

# -----------------------------------------------------------------------------------
# 1. 初始化会话状态 & 回放历史
# -----------------------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

# -----------------------------------------------------------------------------------
# 2. 工具函数：解析用户句子 ➜ 偏好
# -----------------------------------------------------------------------------------
def parse_user_query(query: str) -> dict:
    """从用户句子里提取预算数字和简单关键词"""
    prefs = {"budget": None, "keywords": []}

    m = re.search(r"\$?(\d{2,5})", query)
    if m:
        prefs["budget"] = int(m.group(1))

    for kw in ["red", "blue", "green", "dress", "shoe", "tech", "bag"]:
        if kw in query.lower():
            prefs["keywords"].append(kw)
    return prefs

# -----------------------------------------------------------------------------------
# 3. 规则过滤（预算 / 关键词）
# -----------------------------------------------------------------------------------
def rule_based_recommend(inv: pd.DataFrame, prefs: dict, top_k: int = 5):
    df = inv.copy()

    # 3-1 预算
    if prefs["budget"] is not None and "price" in df.columns:
        df = df[df["price"] <= prefs["budget"]]

    # 3-2 关键词
    if prefs["keywords"]:
        pattern = "|".join(prefs["keywords"])
        cols = [c for c in df.columns if c in ("name", "category", "color")]
        if cols:
            df = df[
                df[cols]
                .astype(str)
                .apply(lambda r: r.str.contains(pattern, case=False))
                .any(axis=1)
            ]

    return df.head(top_k)

# -----------------------------------------------------------------------------------
# 4. (可选) OpenAI LLM 推荐兜底
# -----------------------------------------------------------------------------------
def llm_recommend(inv: pd.DataFrame, query: str, top_k: int = 3):
    sample = inv.head(20).to_dict("records")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "Recommend up to 3 products from the inventory JSON that best match the user query. "
                "Respond in markdown list format: **Name – $Price** newline Reason."
            ),
        },
        {
            "role": "user",
            "content": f"Inventory JSON: {json.dumps(sample)}\n\nUser query: {query}",
        },
    ]
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            timeout=20,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"*LLM fallback failed*: {e}"

# -----------------------------------------------------------------------------------
# 5. 主对话逻辑
# -----------------------------------------------------------------------------------
user_msg = st.chat_input("Tell me what you need (e.g. color, budget, category)")
if user_msg:
    # 显示并存入历史
    st.session_state.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

    # 5-1 侧边栏价格区间过滤
    inv_for_filter = inventory
    if price_min is not None:
        inv_for_filter = inv_for_filter[
            (inv_for_filter["price"] >= price_min) & (inv_for_filter["price"] <= price_max)
        ]

    # 5-2 解析偏好 ➜ 规则过滤
    prefs = parse_user_query(user_msg)
    filtered = rule_based_recommend(inv_for_filter, prefs, top_k=3)

    # 5-3 如果规则过滤为空，则尝试 TF-IDF 推荐
    if filtered.empty:
        tfidf_df = get_recommendations(user_msg, inv_for_filter, k=3)
        if not tfidf_df.empty:
            filtered = tfidf_df

    # 5-4 生成回复
    if not filtered.empty:
        lines = []
        for _, row in filtered.iterrows():
            reason = []
            if prefs["budget"] and "price" in row and row["price"] <= prefs["budget"]:
                reason.append("within budget")
            if prefs["keywords"]:
                reason.append("matches keyword")
            lines.append(
                f"**{row['name']} – ${row['price']}**  \n"
                f"Reason: {', '.join(reason) or 'top pick'}"
            )
        reply_md = "## Top picks\n\n" + "\n\n".join(lines)
    else:
        reply_md = llm_recommend(inv_for_filter, user_msg) if OPENAI_KEY_AVAILABLE else \
            "Sorry, I couldn't find any matching products."

    # 5-5 输出并记录
    st.session_state.chat.append(("assistant", reply_md))
    st.chat_message("assistant").write(reply_md)