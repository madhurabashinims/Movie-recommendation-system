import numpy as np
from collections import defaultdict
from backend.recommender import recommend_movies_multimodal


def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if relevant else 0


def ndcg_at_k(recommended, relevant, k):
    dcg = 0
    for i, movie in enumerate(recommended[:k]):
        if movie in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0


def evaluate_recommender(context, k=10):

    train_ratings = context["ratings"]
    test_ratings = context["test_ratings"]

    users = test_ratings.userId.unique()

# Limit number of users for faster evaluation
    max_users = 1000

    if len(users) > max_users:
        users = np.random.choice(users, size=max_users, replace=False)

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    print("Evaluating users:", len(users))

    for user in users:

        relevant_movies = test_ratings[
            test_ratings.userId == user
        ].movieId.tolist()

        if len(relevant_movies) == 0:
            continue

        try:
            recs = recommend_movies_multimodal(
                user,
                context,
                top_n=k
            )

            recommended_movies = recs.movieId.tolist()

        except Exception:
            continue

        precision_scores.append(
            precision_at_k(recommended_movies, relevant_movies, k)
        )

        recall_scores.append(
            recall_at_k(recommended_movies, relevant_movies, k)
        )

        ndcg_scores.append(
            ndcg_at_k(recommended_movies, relevant_movies, k)
        )

    results = {
        "Precision@K": np.mean(precision_scores),
        "Recall@K": np.mean(recall_scores),
        "NDCG@K": np.mean(ndcg_scores)
    }

    print("\nEvaluation Results")
    print("-------------------")
    print(f"Precision@{k}: {results['Precision@K']:.4f}")
    print(f"Recall@{k}: {results['Recall@K']:.4f}")
    print(f"NDCG@{k}: {results['NDCG@K']:.4f}")

    return results