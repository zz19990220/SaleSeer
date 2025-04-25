# 💬 Chatbot SaleSeer

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-SaleSeer.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
## Setup (virtual environment)

These steps let anyone clone the repo and run **SaleSeer** locally in an isolated Python environment.

```bash
# 1) create and activate virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows users: venv\Scripts\activate

# 2) install all required packages
pip install -r requirements.txt

# 3) start the Streamlit app
streamlit run streamlit_app.py