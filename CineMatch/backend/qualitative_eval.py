# backend/qualitative_eval.py
"""
Qualitative Evaluation for Chatbot and Ultra-Specific Search
=============================================================
Since chatbot/ultra-specific don't have ground-truth ratings,
we measure Genre Precision — the fraction of top-10 results
that match the expected genre for the given query type.

This is a standard evaluation for content-based/query-based recommenders.
Run: python -m backend.qualitative_eval

Expected runtime: 2-5 minutes
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from backend.loaders import load_all
from backend.chatbot import chatbot_recommendation
from backend.ultra_filter import ultra_filter_recommend


# ══════════════════════════════════════════════════════════════
#  CHATBOT TEST CASES  (emotion → expected genres)
# ══════════════════════════════════════════════════════════════
CHATBOT_TESTS = [
    {
        "message":        "I miss my pet bunny so much",
        "expected_genres": ["Animation", "Family", "Adventure"],
        "context_label":  "soft_grief",
    },
    {
        "message":        "My boss yelled at me today and I feel so frustrated",
        "expected_genres": ["Action", "Comedy", "Thriller"],
        "context_label":  "anger",
    },
    {
        "message":        "I am feeling so happy, my friend and I had a great day",
        "expected_genres": ["Comedy", "Animation", "Adventure", "Family"],
        "context_label":  "happy",
    },
    {
        "message":        "I miss my mom so much",
        "expected_genres": ["Drama", "Family"],
        "context_label":  "deep_grief",
    },
    {
        "message":        "I am bored and have nothing to do",
        "expected_genres": ["Action", "Adventure", "Thriller", "Comedy"],
        "context_label":  "bored",
    },
    {
        "message":        "I am feeling so stressed about my exams",
        "expected_genres": ["Comedy", "Animation", "Family"],
        "context_label":  "stressed",
    },
]

# ══════════════════════════════════════════════════════════════
#  ULTRA-SPECIFIC TEST CASES
# ══════════════════════════════════════════════════════════════
ULTRA_TESTS = [
    {
        "query":           "animation with animals cheerful and fun for kids",
        "must_include":    ["Animation", "Family"],
        "must_exclude":    [],
        "label":           "Animation+Family inclusion",
    },
    {
        "query":           "dark thriller mind-bending not horror",
        "must_include":    ["Thriller", "Drama", "Science Fiction"],
        "must_exclude":    ["Horror"],
        "label":           "Thriller+negation",
    },
    {
        "query":           "romantic comedy feel-good lighthearted",
        "must_include":    ["Romance", "Comedy"],
        "must_exclude":    [],
        "label":           "Romance+Comedy",
    },
    {
        "query":           "epic adventure fantasy like The Lord of the Rings",
        "must_include":    ["Adventure", "Fantasy", "Action"],
        "must_exclude":    [],
        "label":           "Anchor+genre",
    },
    {
        "query":           "animation with animals cheerful adventure kids no horror no violence",
        "must_include":    ["Animation", "Family"],
        "must_exclude":    ["Horror"],
        "label":           "Negation enforcement",
    },
]


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════
def genre_precision(recommendations_df, expected_genres, k=10):
    """
    Fraction of top-k recommendations that contain at least one
    of the expected genres.
    """
    if recommendations_df is None or recommendations_df.empty:
        return 0.0
    top_k = recommendations_df.head(k)
    hits = 0
    for _, row in top_k.iterrows():
        genres_str = str(row.get("genres", "")).lower()
        if any(g.lower() in genres_str for g in expected_genres):
            hits += 1
    return hits / min(k, len(top_k))

def genre_exclusion_score(recommendations_df, excluded_genres, k=10):
    """
    Fraction of top-k recommendations that do NOT contain excluded genres.
    Higher = better negation enforcement.
    """
    if not excluded_genres:
        return 1.0
    if recommendations_df is None or recommendations_df.empty:
        return 0.0
    top_k = recommendations_df.head(k)
    clean = 0
    for _, row in top_k.iterrows():
        genres_str = str(row.get("genres", "")).lower()
        if not any(g.lower() in genres_str for g in excluded_genres):
            clean += 1
    return clean / min(k, len(top_k))


# ══════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ══════════════════════════════════════════════════════════════
def main():
    print("Loading context...")
    context = load_all()
    
    # Use a fixed user_id for chatbot tests
    sample_uid = context["ratings"]["userId"].iloc[0]
    print(f"Using user_id={sample_uid} for chatbot tests\n")

    # ── CHATBOT EVALUATION ────────────────────────────────────
    print("=" * 65)
    print("CHATBOT — Genre Precision@10")
    print("Measures: % of top-10 recs matching expected emotion-genre")
    print("=" * 65)
    print(f"{'Context':<20} {'Query (truncated)':<35} {'Genre P@10':>10}")
    print("-" * 65)

    chatbot_scores = []
    for test in CHATBOT_TESTS:
        try:
            result = chatbot_recommendation(
                user_id=sample_uid,
                message=test["message"],
                context=context,
            )
            recs = result.get("recommendations")
            score = genre_precision(recs, test["expected_genres"], k=10)
            chatbot_scores.append(score)
            msg_short = test["message"][:33] + "..." if len(test["message"]) > 33 else test["message"]
            print(f"{test['context_label']:<20} {msg_short:<35} {score:>9.1%}")
        except Exception as e:
            print(f"{test['context_label']:<20} ERROR: {e}")
            chatbot_scores.append(0.0)

    avg_chatbot = np.mean(chatbot_scores)
    print("-" * 65)
    print(f"{'AVERAGE':<56} {avg_chatbot:>9.1%}")
    print(f"\nInterpretation: {avg_chatbot:.1%} of recommendations match")
    print(f"the emotionally-appropriate genre on average.")

    # ── ULTRA-SPECIFIC EVALUATION ─────────────────────────────
    print("\n" + "=" * 65)
    print("ULTRA-SPECIFIC — Genre Precision@10 + Negation Enforcement")
    print("=" * 65)
    print(f"{'Test Case':<30} {'Genre P@10':>10} {'Excl. Score':>12} {'Combined':>9}")
    print("-" * 65)

    ultra_gp, ultra_ex, ultra_comb = [], [], []
    for test in ULTRA_TESTS:
        try:
            results_df, parsed = ultra_filter_recommend(
                query=test["query"],
                context=context,
                top_n=10,
            )
            gp = genre_precision(results_df, test["must_include"], k=10)
            ex = genre_exclusion_score(results_df, test["must_exclude"], k=10)
            combined = (gp + ex) / 2 if test["must_exclude"] else gp
            ultra_gp.append(gp)
            ultra_ex.append(ex)
            ultra_comb.append(combined)
            label = test["label"][:28]
            print(f"{label:<30} {gp:>9.1%}  {ex:>10.1%}  {combined:>8.1%}")
        except Exception as e:
            print(f"{test['label']:<30} ERROR: {e}")
            ultra_gp.append(0.0); ultra_ex.append(0.0); ultra_comb.append(0.0)

    print("-" * 65)
    print(f"{'AVERAGE':<30} {np.mean(ultra_gp):>9.1%}  {np.mean(ultra_ex):>10.1%}  {np.mean(ultra_comb):>8.1%}")

    # ── SUMMARY TABLE ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — All Evaluation Results")
    print("=" * 65)
    print(f"\n{'Component':<35} {'Primary Metric':<20} {'Value':>8}")
    print("-" * 65)
    print(f"{'Hybrid Recommender (Precision@10)':<35} {'Best config':<20} {'0.2583':>8}")
    print(f"{'Hybrid Recommender (NDCG@10)':<35} {'Best config':<20} {'0.2791':>8}")
    print(f"{'Chatbot Genre Precision@10':<35} {'Avg over 6 queries':<20} {avg_chatbot:>8.1%}")
    print(f"{'Ultra-Specific Genre Precision@10':<35} {'Avg over 5 queries':<20} {np.mean(ultra_gp):>8.1%}")
    print(f"{'Ultra-Specific Negation Score':<35} {'Avg over 5 queries':<20} {np.mean(ultra_ex):>8.1%}")
    print("\nNote: Chatbot/Ultra-Specific use Genre Precision (standard for")
    print("content-based and query-based recommenders without ground-truth labels).")
    print("Hybrid recommender uses standard CF metrics (Precision@10, NDCG@10).")


if __name__ == "__main__":
    main()