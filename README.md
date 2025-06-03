# 🛒 SaleSeer – AI Product Recommender

A lightweight Streamlit demo that recommends products from your own inventory CSV (or a sample inventory) and—if you set an `OPENAI_API_KEY`—can fall back to GPT-4o for extra suggestions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://saleseer-dev771k8c1kr.streamlit.app/)

---

## 🚀 Quick Start

```bash
# 1) create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) (optional) enable LLM fallback
export OPENAI_API_KEY=<your key>

# 4) run the app
streamlit run streamlit_app.py