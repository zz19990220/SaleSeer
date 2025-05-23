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

# ---------- 本地解析函数（已在 parser/ 里） ----------
from inv_parser.inventory_parser import load_sample_inventory, load_csv
# -----------------------------------------------------

st.set_page_config(page_title="SaleSeer", page_icon="🛒")
st.title("🛒 SaleSeer – AI Product Recommender")
st.write(
    "Upload **your** inventory CSV or click *Load sample inventory* to try the demo."
)

# --------------------------------------------
# 0. Sidebar – Inventory: 缓存或首次加载
# --------------------------------------------
with st.sidebar:
    
    # 0-1 如果之前已经加载过，就直接用缓存
    if "inventory" in st.session_state:
        inventory = st.session_state.inventory
        st.success(f"{len(inventory)} products ready (cached)")
    else:
        # 0-2 首次：按钮 or 上传
        if st.button("Load sample inventory"):
            inventory = load_sample_inventory()
            st.session_state.inventory = inventory
            st.success(f"{len(inventory)} sample products loaded")
        else:
            csv_file = st.file_uploader("Upload inventory CSV", type="csv")
            if csv_file:
                inventory = load_csv(csv_file)
                st.session_state.inventory = inventory
                st.success(f"{len(inventory)} products loaded")
            else:
                st.info(
                    "Please upload an inventory CSV to start, or click **Load sample inventory** above."
                )
                st.stop()  # 本次 run 没有任何库存数据 → 中断

    # 侧边栏预览 & Reset
    with st.expander("Preview inventory"):
        st.dataframe(inventory.head())

    if st.button("Reset chat & inventory"):
        st.session_state.clear()
        st.experimental_rerun()

# --------------------------------------------
# 1. 初始化会话状态
# --------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# 重播历史
for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

# --------------------------------------------
# 2. 工具函数：解析用户句子 ➜ 偏好
# --------------------------------------------
def parse_user_query(query: str) -> dict:
    """
    Super-light parser:
    - 预算：捕获 $或数字
    - 颜色 / 类别：用简单关键词
    """
    prefs = {"budget": None, "keywords": []}

    # 预算
    m = re.search(r"\$?(\d{2,5})", query)
    if m:
        prefs["budget"] = int(m.group(1))

    # 关键词（可根据你的库存字段再延展）
    for kw in ["red", "blue", "green", "dress", "shoe", "tech", "bag"]:
        if kw in query.lower():
            prefs["keywords"].append(kw)

    return prefs

# --------------------------------------------
# 3. 规则过滤（匹配预算 / 关键词）
# --------------------------------------------
def rule_based_recommend(inv: pd.DataFrame, prefs: dict, top_k: int = 5):
    df = inv.copy()

    # 预算过滤
    if prefs["budget"] is not None and "price" in df.columns:
        df = df[df["price"] <= prefs["budget"]]

    # 关键词过滤（在 name 或 category 字段里搜）
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

# --------------------------------------------
# 4. (可选) OpenAI LLM 推荐兜底
# --------------------------------------------
def llm_recommend(inv: pd.DataFrame, query: str, top_k: int = 3):
    """
    将前 20 条商品示例 + 用户需求一起发给 GPT，
    让它直接返回 markdown 列表。
    """
    sample = inv.head(20).to_dict("records")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant. "
            "Recommend up to 3 products from the inventory JSON that best match the user query. "
            "Respond in markdown list format: **Name – $Price** newline Reason.",
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

# --------------------------------------------
# 5. 主对话逻辑
# --------------------------------------------
user_msg = st.chat_input("Tell me what you need (e.g. color, budget, category)")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

    prefs = parse_user_query(user_msg)
    filtered = rule_based_recommend(inventory, prefs, top_k=3)

    if not filtered.empty:  # 用规则推荐成功
        lines = []
        for idx, row in filtered.iterrows():
            reason = []
            if prefs["budget"]:
                reason.append("within budget")
            if prefs["keywords"]:
                reason.append("matches keyword")
            lines.append(
                f"{idx + 1}. **{row['name']} – ${row['price']}**  \n"
                f"   Reason: {', '.join(reason) or 'top pick'}"
            )
        reply_md = "**Top picks**\n\n" + "\n\n".join(lines)
    else:  # 兜底：调用 LLM
        if OPENAI_KEY_AVAILABLE:
            reply_md = llm_recommend(inventory, user_msg)
        else:
            reply_md = "Sorry, I couldn't find any matching products."

    st.session_state.chat.append(("assistant", reply_md))
    st.chat_message("assistant").write(reply_md)