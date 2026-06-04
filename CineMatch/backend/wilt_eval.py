# backend/wilt_eval.py
"""
Evaluation for "Will I Like This?" Tab
========================================
Uses actual user ratings to measure binary prediction accuracy.

Method:
  For each sampled user, pick:
    - 10 movies they rated >= 4.0 (positive — should predict "like")
    - 10 movies they rated <= 2.0 (negative — should predict "unlikely")
  Run predict_user_like_movie() on all 20.
  Measure: Accuracy, Precision, Recall, F1.

This is a genuine holdout evaluation — movies come from the test split
so the model has not seen them during training.

Run: python -m backend.wilt_eval
Expected runtime: 5-10 minutes
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backend.loaders import load_all
from backend.train_test_split import split_ratings
from backend.predictors import predict_user_like_movie


def evaluate_wilt(context, test_ratings, n_users=100, n_per_class=5):
    """
    Evaluate Will I Like This? using held-out test ratings.
    
    For each user:
    - Sample n_per_class positive movies (rating >= 4.0) from test set
    - Sample n_per_class negative movies (rating <= 2.0) from test set
    - Run predictor on each
    - Check if prediction matches ground truth
    """
    movies   = context["movies"].drop_duplicates("movieId").reset_index(drop=True)
    
    # Build movie text lookup: movieId -> text description
    def build_movie_text(row):
        title    = str(row.get("title", "") or "")
        genres   = str(row.get("genres", "") or "").replace("|", " ")
        overview = str(row.get("overview", "") or "")[:300]
        tagline  = str(row.get("tagline", "") or "")
        return f"{title} {genres} {overview} {tagline}".strip()
    
    movie_text_map = {
        row["movieId"]: build_movie_text(row)
        for _, row in movies.iterrows()
    }
    
    # Sample users
    np.random.seed(42)
    all_users = test_ratings["userId"].unique()
    sample_users = np.random.choice(
        all_users, size=min(n_users, len(all_users)), replace=False
    )
    
    results = []
    tp = fp = tn = fn = 0
    
    print(f"  Evaluating {len(sample_users)} users ({n_per_class} pos + {n_per_class} neg each)...")
    
    for i, uid in enumerate(sample_users):
        user_test = test_ratings[test_ratings["userId"] == uid]
        
        positives = user_test[user_test["rating"] >= 4.0]["movieId"].tolist()
        negatives = user_test[user_test["rating"] <= 2.0]["movieId"].tolist()
        
        # Need at least some of each
        if len(positives) < 2 or len(negatives) < 2:
            continue
        
        pos_sample = np.random.choice(
            positives, size=min(n_per_class, len(positives)), replace=False
        )
        neg_sample = np.random.choice(
            negatives, size=min(n_per_class, len(negatives)), replace=False
        )
        
        for mid in pos_sample:
            text = movie_text_map.get(mid, "unknown movie")
            if not text or text == "unknown movie":
                continue
            try:
                pred = predict_user_like_movie(
                    user_id=uid, movie_text=text,
                    context=context, uploaded_poster=None
                )
                predicted_like = pred["verdict"] in ("like", "maybe")
                if predicted_like:
                    tp += 1
                else:
                    fn += 1
                results.append({
                    "uid": uid, "mid": mid, "true": 1,
                    "predicted": int(predicted_like),
                    "score": pred["final_score"],
                    "verdict": pred["verdict"]
                })
            except Exception:
                continue
        
        for mid in neg_sample:
            text = movie_text_map.get(mid, "unknown movie")
            if not text or text == "unknown movie":
                continue
            try:
                pred = predict_user_like_movie(
                    user_id=uid, movie_text=text,
                    context=context, uploaded_poster=None
                )
                predicted_like = pred["verdict"] in ("like", "maybe")
                if predicted_like:
                    fp += 1
                else:
                    tn += 1
                results.append({
                    "uid": uid, "mid": mid, "true": 0,
                    "predicted": int(predicted_like),
                    "score": pred["final_score"],
                    "verdict": pred["verdict"]
                })
            except Exception:
                continue
        
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(sample_users)} users done")
    
    total = tp + fp + tn + fn
    if total == 0:
        print("  ERROR: No predictions made.")
        return {}
    
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Score distribution
    df = pd.DataFrame(results)
    pos_scores = df[df["true"] == 1]["score"].values
    neg_scores = df[df["true"] == 0]["score"].values
    
    return {
        "accuracy":   round(accuracy, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total":      total,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "df":         df,
    }


def main():
    print("Loading context...")
    context     = load_all()
    _, test_ratings = None, None
    
    if "test_ratings" in context:
        test_ratings = context["test_ratings"]
        print(f"  Using context test_ratings: {len(test_ratings)} rows")
    else:
        from backend.config import RATINGS_PATH
        import pandas as pd
        all_ratings  = pd.read_csv(RATINGS_PATH)
        from backend.train_test_split import split_ratings
        _, test_ratings = split_ratings(all_ratings)
        print(f"  Reloaded test_ratings: {len(test_ratings)} rows")
    
    print("\nRunning Will I Like This? evaluation...")
    metrics = evaluate_wilt(context, test_ratings, n_users=150, n_per_class=5)
    
    if not metrics:
        return
    
    # ── Print results ─────────────────────────────────────────
    print("\n" + "="*55)
    print("WILL I LIKE THIS? — EVALUATION RESULTS")
    print("="*55)
    print(f"  Total predictions : {metrics['total']}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  : {metrics['tp']}  (correctly predicted LIKE)")
    print(f"    True Negatives  : {metrics['tn']}  (correctly predicted UNLIKELY)")
    print(f"    False Positives : {metrics['fp']}  (predicted LIKE but user didn't)")
    print(f"    False Negatives : {metrics['fn']}  (predicted UNLIKELY but user did like)")
    
    # Score separation check
    if len(metrics['pos_scores']) > 0 and len(metrics['neg_scores']) > 0:
        print(f"\n  Score Separation:")
        print(f"    Avg score for movies user DID like   : {metrics['pos_scores'].mean():.4f}")
        print(f"    Avg score for movies user DIDN'T like: {metrics['neg_scores'].mean():.4f}")
        gap = metrics['pos_scores'].mean() - metrics['neg_scores'].mean()
        print(f"    Gap (positive signal)                : {gap:+.4f}")
        if gap > 0:
            print(f"    ✓ Model correctly assigns higher scores to liked movies")
        else:
            print(f"    ✗ Warning: scores not well separated")
    
    # ── Generate graph ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: bar chart of metrics
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_vals  = [metrics["accuracy"], metrics["precision"],
                    metrics["recall"], metrics["f1"]]
    colors_bar = ["#2563eb", "#0891b2", "#16a34a", "#d97706"]
    bars = ax1.bar(metric_names, metric_vals, color=colors_bar, alpha=0.85, width=0.5)
    for bar, val in zip(bars, metric_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.set_title("Will I Like This? — Prediction Metrics\n(Binary Classification on Test Ratings)",
                  fontsize=11, fontweight='bold')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: score distribution
    if len(metrics['pos_scores']) > 0 and len(metrics['neg_scores']) > 0:
        ax2.hist(metrics['pos_scores'], bins=20, alpha=0.7, color='#16a34a',
                 label=f"Liked (rated ≥4.0)\nn={len(metrics['pos_scores'])}")
        ax2.hist(metrics['neg_scores'], bins=20, alpha=0.7, color='#dc2626',
                 label=f"Disliked (rated ≤2.0)\nn={len(metrics['neg_scores'])}")
        ax2.set_xlabel("Predicted Compatibility Score")
        ax2.set_ylabel("Count")
        ax2.set_title("Score Distribution: Liked vs Disliked Movies",
                      fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig('graph5_wilt.png', bbox_inches='tight')
    plt.close()
    print("\n  graph5_wilt.png saved to D:\\MINI PROJECT\\")
    
    print("\n" + "="*55)
    print("INTERPRETATION:")
    acc_pct = metrics['accuracy'] * 100
    if acc_pct >= 65:
        print(f"  Accuracy of {acc_pct:.1f}% is above the 50% random baseline,")
        print(f"  confirming the predictor distinguishes liked from disliked movies.")
    else:
        print(f"  Accuracy of {acc_pct:.1f}% — scores are close to random.")
        print(f"  The text-only similarity has limited discriminative power.")
        print(f"  Note: without uploaded poster, only text features are used.")
    print(f"\n  Key insight: the predictor uses text similarity between the")
    print(f"  movie's description and the user's viewing history profile.")
    print(f"  When a poster is uploaded, visual features (alpha=0.7) are added.")


if __name__ == "__main__":
    main()