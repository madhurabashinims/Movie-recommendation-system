# backend/recommender.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize(x):
    return (x - x.min()) / (np.ptp(x) + 1e-8)


def build_user_vector(user_id, ratings, embeddings, movie_id_to_index):
    seen = ratings[ratings.userId == user_id]["movieId"]

    indices = [
        movie_id_to_index[mid]
        for mid in seen
        if mid in movie_id_to_index and
           np.linalg.norm(embeddings[movie_id_to_index[mid]]) > 0
    ]

    if not indices:
        return None

    return embeddings[indices].mean(axis=0)


def recommend_movies_multimodal(
    user_id,
    context,
    alpha=0.6,   # NCF
    beta=0.25,    # Text
    gamma=0.15,   # Poster
    top_n=10
):
    movies = context["movies"]
    ratings = context["ratings"]
    ncf_model = context["ncf_model"]
    user_encoder = context["user_encoder"]
    movie_encoder = context["movie_encoder"]
    text_embeddings = context["text_embeddings"]
    poster_embeddings = context["poster_embeddings"]

    movie_id_to_index = dict(zip(movies.movieId, movies.index))

    # -------- Encode user --------
    user_encoded = user_encoder.transform([user_id])[0]

    seen_movies = set(ratings[ratings.userId == user_id]["movieId"])

    candidate_movies = [
        m for m in movies.movieId
        if m not in seen_movies and m in movie_encoder.classes_
    ]

    candidate_encoded = movie_encoder.transform(candidate_movies)

    # -------- NCF --------
    user_input = np.full(len(candidate_encoded), user_encoded)

    ncf_scores = ncf_model.predict(
        [user_input, candidate_encoded],
        verbose=0
    ).flatten()
    ncf_scores = normalize(ncf_scores)

    # -------- TEXT --------
    user_text_vec = build_user_vector(
        user_id, ratings, text_embeddings, movie_id_to_index
    )

    candidate_indices = [movie_id_to_index[m] for m in candidate_movies]

    if user_text_vec is not None:
        text_scores = cosine_similarity(
            user_text_vec.reshape(1, -1),
            text_embeddings[candidate_indices]
        )[0]
        text_scores = normalize(text_scores)
    else:
        text_scores = np.zeros(len(candidate_movies))

    # -------- POSTER --------
    user_poster_vec = build_user_vector(
        user_id, ratings, poster_embeddings, movie_id_to_index
    )

    if user_poster_vec is not None:
        poster_scores = cosine_similarity(
            user_poster_vec.reshape(1, -1),
            poster_embeddings[candidate_indices]
        )[0]
        poster_scores = normalize(poster_scores)
    else:
        poster_scores = np.zeros(len(candidate_movies))

    # -------- FINAL --------
    final_scores = (
        alpha * ncf_scores +
        beta * text_scores +
        gamma * poster_scores
    )

    top_idx = np.argsort(final_scores)[::-1][:top_n]
    rec_ids = [candidate_movies[i] for i in top_idx]

    return movies[movies.movieId.isin(rec_ids)][
    ["movieId", "title", "genres", "poster_path"]
]

