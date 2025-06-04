# recommendation/engine.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _prepare_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine product name and description to generate text corpus"""
    text_cols = ["name", "description"]
    return df[text_cols].fillna("").agg(" ".join, axis=1)

def get_recommendations(query: str, inventory_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Return top k products with highest similarity to user query
    Only returns items with similarity > 0.05 to avoid unrelated matches
    """
    # 1) Prepare corpus
    corpus = _prepare_corpus(inventory_df)

    # 2) Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    item_vecs = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])

    # 3) Cosine similarity
    scores = cosine_similarity(query_vec, item_vecs).flatten()

    # 4) Filter out low similarity scores (threshold = 0.05)
    similarity_threshold = 0.05
    valid_indices = np.where(scores > similarity_threshold)[0]
    
    if len(valid_indices) == 0:
        # No meaningful matches found
        return pd.DataFrame(columns=list(inventory_df.columns) + ["similarity"])
    
    # 5) Sort by similarity and take top k from valid matches
    valid_scores = scores[valid_indices]
    sorted_idx = valid_indices[np.argsort(valid_scores)[::-1]]
    
    # Take at most k results
    top_idx = sorted_idx[:min(k, len(sorted_idx))]
    
    results = inventory_df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]
    
    # Round similarity to 3 decimal places for cleaner display
    results["similarity"] = results["similarity"].round(3)
    
    return results.reset_index(drop=True)