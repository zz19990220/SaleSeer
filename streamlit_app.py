import streamlit as st
import pandas as pd
from textwrap import dedent

st.set_page_config(page_title="SaleSeer")

st.title("🛒 SaleSeer – AI Product Recommender")
st.write(
    "Upload **your** inventory CSV or click *Load sample inventory* to try the demo instantly."
)

# --------------------------------------------
# 1. Inventory: upload or load sample
# --------------------------------------------
if st.sidebar.button("Load sample inventory"):
    inventory = pd.read_csv("assets/sample_inventory.csv")
    st.sidebar.success(f"{len(inventory)} sample products loaded")
else:
    csv_file = st.sidebar.file_uploader("Upload inventory CSV", type="csv")
    if not csv_file:
        st.sidebar.info("Please upload an inventory file to start")
        st.stop()
    inventory = pd.read_csv(csv_file)
    st.sidebar.success(f"{len(inventory)} products loaded")

with st.sidebar.expander("Preview inventory"):
    st.dataframe(inventory.head())

if st.sidebar.button("Reset chat"):
    st.session_state.chat = []
    st.experimental_rerun()

# --------------------------------------------
# 2. Chat history
# --------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

# --------------------------------------------
# 3. User input  ➜  generate mock response
# --------------------------------------------
user_msg = st.chat_input("Tell me what you need (e.g. color, budget, category)")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

    # Pick up to 3 products as dummy recommendations
    top_n = min(3, len(inventory))
    items = inventory.head(top_n).reset_index(drop=True)

    lines = []
    for idx, row in items.iterrows():
        reason = "matches your request" if idx == 0 else "alternative option"
        lines.append(
            f"{idx+1}. *{row['name']}* – ${row['price']}  \n"
            f"   Reason: {reason}."
        )

    reply = dedent("""\
        **Top picks**

        """ + "\n\n".join(lines)
    ).strip()

    st.session_state.chat.append(("assistant", reply))
    st.chat_message("assistant").write(reply)