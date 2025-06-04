# recommendation/gpt_engine.py

import os
from typing import List, Dict, Tuple
import openai  # type: ignore
import logging

# Configure OpenRouter API
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_recommendation(query: str, top_k_products: List[Dict]) -> Tuple[str, str]:
    """
    Use GPT to wrap TF-IDF filtered Top-K products into natural language recommendation.
    
    Parameters:
    - query: User's original input, e.g. "shoes under 500"
    - top_k_products: Product list [{"name": ..., "price": ...}, ...]
    
    Returns:
    - Tuple of (GPT generated recommendation string, model_used)
    """

    # 2) Compile brief product information
    product_lines = []
    for item in top_k_products:
        name = item.get("name", "Unnamed Product")
        price = item.get("price", "N/A")
        product_lines.append(f"- {name} (${price})")
    product_text = "\n".join(product_lines)

    # 3) Construct Prompt (in English for consistency with the app)
    prompt = (
        f"User is looking for: {query}\n\n"
        f"Here are the candidate products selected from inventory:\n{product_text}\n\n"
        f"As a friendly and professional shopping advisor, write a brief English recommendation "
        f"to help the user choose the most suitable products from the above options."
    )

    # 4) Try GPT-4o-mini first, fallback to GPT-3.5-turbo
    models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    for i, model in enumerate(models_to_try):
        try:
            logger.info(f"Attempting to use {model} for recommendation generation")
            resp = openai.ChatCompletion.create(  # type: ignore
                model=model,
                messages=[
                    {"role": "system", "content": "You are a friendly and professional shopping advisor who excels at giving personalized recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
                timeout=20
            )
            recommendation = resp.choices[0].message["content"].strip()  # type: ignore
            logger.info(f"Successfully generated recommendation using {model}")
            return recommendation, model
            
        except openai.error.OpenAIError as e:  # type: ignore
            # Check for specific error types that warrant fallback
            error_code = getattr(e, 'code', None)
            if error_code in ["insufficient_quota", "invalid_request_error", "rate_limit_exceeded"] or \
               "quota" in str(e).lower() or "rate limit" in str(e).lower():
                
                logger.warning(f"OpenAI quota/rate limit error with {model}: {e}")
                
                # If this was the last model to try, return error
                if i == len(models_to_try) - 1:
                    logger.error(f"All models exhausted. Final error: {e}")
                    return f"*GPT recommendation failed: {e}*", "error"
                
                # Otherwise, continue to next model (fallback)
                logger.info(f"Falling back from {model} to {models_to_try[i+1]}")
                continue
            else:
                # For other errors, don't retry
                logger.error(f"Non-recoverable OpenAI error with {model}: {e}")
                return f"*GPT recommendation failed: {e}*", "error"
                
        except Exception as e:
            # For non-OpenAI errors, don't retry
            logger.error(f"Unexpected error with {model}: {e}")
            return f"*GPT recommendation failed: {e}*", "error"
    
    # Should not reach here, but just in case
    logger.error("All models exhausted without clear fallback path")
    return "*GPT recommendation failed: All models exhausted*", "error"