# recommendation/gpt_engine.py

import os
import json

# For OpenAI v0.28.0 legacy API
try:
    import openai  # type: ignore
except ImportError:
    openai = None

def generate_recommendation(query: str, top_k_list: list) -> tuple:
    """
    Given:
      - query: the user's original free-text query
      - top_k_list: a list of dicts like [{"name": ..., "price": ...}, ...]
    Returns:
      - (reply_text, model_used), where model_used is either "gpt-4o-mini" or "gpt-3.5-turbo" or "error".
    This function tries to call the OpenRouter Chat Completion endpoint. If it fails (quota, network, missing key),
    it falls back to returning a static message and model_used="error".
    """
    # Check that we have the OpenAI client available
    if openai is None or not hasattr(openai, "api_key") or openai.api_key is None:
        return ("*OpenAI service unavailable. Cannot generate GPT recommendation.*", "error")

    # Build a very simple prompt: "Here are 3 items. Write a short, friendly recommendation."
    # We include the prices and names in JSON.
    try:
        snippet_json = json.dumps(top_k_list, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": "You are a friendly, professional AI shopping assistant. Given a few product names and prices, write a short paragraph recommending the best match for the user, based on their original query."
            },
            {
                "role": "user",
                "content": f"User query: {query}\nInventory snippet (top 3 as JSON): {snippet_json}"
            }
        ]

        # Try GPT-4o-mini first using custom OpenRouter wrapper
        try:
            # Import the requests-based wrapper for reliable OpenRouter calls
            import sys
            sys.path.append('..')
            from openai_wrapper import call_openrouter_chat
            
            # Try GPT-4o-mini first
            response_text, model_used = call_openrouter_chat(messages, model="gpt-4o-mini", temperature=0.7)
            if model_used != "error":
                return (response_text, model_used)
            else:
                # If that failed, try gpt-3.5-turbo
                response_text2, model_used2 = call_openrouter_chat(messages, model="gpt-3.5-turbo", temperature=0.7)
                return (response_text2, model_used2)
        except Exception as e1:
            return (f"*GPT recommendation failed: {str(e1)}. The OpenAI key may be invalid or expired.*", "error")

    except Exception as e:
        return (f"*GPT engine encountered an error: {str(e)}*", "error")