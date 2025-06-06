import requests
import json
import random

def call_openrouter_chat(messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=None):
    """
    Direct OpenRouter API call using requests library to avoid OpenAI client compatibility issues.
    Returns (response_text, model_used) or (error_message, "error")
    """
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.app",  # Optional for OpenRouter rankings
        "X-Title": "SaleSeer"  # Optional for OpenRouter rankings
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    try:
        # First try with the requested model
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            return (content, model)
        elif response.status_code == 429 or "quota" in response.text.lower():
            # If quota exceeded on primary model, try fallback
            if model == "gpt-4o-mini":
                payload["model"] = "gpt-3.5-turbo"
                fallback_response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if fallback_response.status_code == 200:
                    result = fallback_response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    return (content, "gpt-3.5-turbo")
            
            return (f"*API quota exceeded. Status: {response.status_code}*", "error")
        elif response.status_code == 401:
            # If authentication fails, provide demo mode response
            return generate_demo_response(messages, model)
        else:
            return (f"*API error. Status: {response.status_code} - {response.text[:200]}*", "error")
            
    except requests.exceptions.Timeout:
        return ("*API request timed out*", "error")
    except Exception as e:
        # If any exception occurs, fall back to demo mode
        return generate_demo_response(messages, model)

def generate_demo_response(messages, model):
    """
    Generate a demo response when the real API is unavailable.
    This provides realistic-looking recommendations for demonstration purposes.
    """
    user_message = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    # Generate contextual demo responses based on the user query
    demo_responses = [
        f"Based on your query '{user_message}', I'd recommend checking out the top-rated products from our TF-IDF analysis. "
        f"These items show strong semantic similarity to your search terms and offer excellent value for money. "
        f"Consider the product features, customer reviews, and price points when making your decision.",
        
        f"Great choice! For '{user_message}', the TF-IDF algorithm has identified some excellent matches. "
        f"I particularly like the balance of features and pricing in these recommendations. "
        f"The similarity scores indicate these products closely match what you're looking for.",
        
        f"Excellent query! The products shown have high semantic relevance to '{user_message}'. "
        f"I'd suggest comparing the key features and reading the descriptions carefully. "
        f"These recommendations represent the best matches from our current inventory based on your search criteria."
    ]
    
    response = random.choice(demo_responses)
    return (f"🎯 **Demo Mode** - {response} *(Note: OpenAI API unavailable, showing demo response)*", "demo-gpt")

# Test function
if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "I need blue headphones under $200"}
    ]
    result, model = call_openrouter_chat(test_messages)
    print(f"Result: {result}")
    print(f"Model: {model}") 