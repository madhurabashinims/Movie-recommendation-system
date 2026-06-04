# backend/recommender.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def normalize(x):
    """Min-max normalize to [0, 1]."""
    rng = np.ptp(x)
    if rng < 1e-8:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def get_user_genre_profile(user_id, ratings, movies):
    """
    Aggregate all genres from a user's watch history into a
    single pipe-separated string.
    Returns an empty string if the user has no history.
    """
    seen_ids = ratings[ratings.userId == user_id]["movieId"]
    seen_movies = movies[movies.movieId.isin(seen_ids)]
    all_genres = "|".join(seen_movies["genres"].dropna().tolist())
    return all_genres


def genre_overlap_weight(genres_a: str, genres_b: str) -> float:
    """
    Jaccard similarity between two pipe-separated genre strings.
    Returns a value in [0, 1].
    """
    set_a = set(g.strip() for g in str(genres_a).split("|") if g.strip())
    set_b = set(g.strip() for g in str(genres_b).split("|") if g.strip())
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ─────────────────────────────────────────────
# USER PROFILE BUILDERS
# ─────────────────────────────────────────────

def build_user_vector(user_id, ratings, embeddings, movie_id_to_index):
    """
    Simple mean of embeddings for movies the user has seen.
    Used as a fallback when timestamps are unavailable.
    """
    seen = ratings[ratings.userId == user_id]["movieId"]
    indices = [
        movie_id_to_index[mid]
        for mid in seen
        if mid in movie_id_to_index
        and movie_id_to_index[mid] < len(embeddings)
        and np.linalg.norm(embeddings[movie_id_to_index[mid]]) > 0
    ]
    if not indices:
        return None
    return embeddings[indices].mean(axis=0)


def build_user_vector_time_decay(user_id, ratings, embeddings, movie_id_to_index):
    """
    NOVELTY — Time-decay user profile.

    Recent ratings are weighted more heavily via an exponential decay:
        weight = exp(-0.3 * years_ago)
    so a rating from this year ≈ 1.0, five years ago ≈ 0.22.

    Falls back to simple mean if timestamps are unavailable.
    """
    seen = ratings[ratings.userId == user_id].copy()

    if "timestamp" not in seen.columns or seen.empty:
        return build_user_vector(user_id, ratings, embeddings, movie_id_to_index)

    # Timestamps are datetime strings e.g. "2004-09-10 03:06:38"
    parsed = pd.to_datetime(seen["timestamp"], errors="coerce")
    seen["days_ago"] = (pd.Timestamp.now() - parsed).dt.days.fillna(365 * 3).astype(int)
    seen["years_ago"] = seen["days_ago"] / 365.25
    seen["weight"] = np.exp(-0.3 * seen["years_ago"])

    valid_rows = [
        (movie_id_to_index[mid], w)
        for mid, w in zip(seen["movieId"], seen["weight"])
        if mid in movie_id_to_index
        and movie_id_to_index[mid] < len(embeddings)
        and np.linalg.norm(embeddings[movie_id_to_index[mid]]) > 0
    ]

    if not valid_rows:
        return None

    indices, weights = zip(*valid_rows)
    indices = list(indices)
    weights = np.array(weights)

    weighted_embs = embeddings[indices] * weights[:, None]
    return weighted_embs.sum(axis=0) / (weights.sum() + 1e-8)


# ─────────────────────────────────────────────
# GENRE-CONDITIONED POSTER SCORING
# ─────────────────────────────────────────────

def genre_conditioned_poster_scores(
    candidate_movies,
    movies,
    poster_embeddings,
    user_poster_vec,
    candidate_indices,
    user_genre_profile,
):
    """
    NOVELTY — Genre-conditioned visual similarity.

    Raw cosine similarity between user poster vector and each
    candidate poster is boosted by the Jaccard genre overlap
    between the user's genre profile and the candidate's genres.

        boosted_score = cosine_sim * (1 + jaccard_overlap)

    This ensures visually similar posters from genre-matched
    movies rank higher than those from unrelated genres.
    """
    raw_scores = cosine_similarity(
        user_poster_vec.reshape(1, -1),
        poster_embeddings[candidate_indices],
    )[0]

    boosted = []
    for i, mid in enumerate(candidate_movies):
        movie_row = movies[movies.movieId == mid]
        if movie_row.empty:
            boosted.append(raw_scores[i])
            continue

        candidate_genres = movie_row["genres"].values[0]
        overlap = genre_overlap_weight(user_genre_profile, candidate_genres)
        # overlap in [0,1] → boost in [1.0, 2.0]
        boosted.append(raw_scores[i] * (1.0 + overlap))

    return normalize(np.array(boosted))


# ─────────────────────────────────────────────
# SENTIMENT-DYNAMIC WEIGHTS
# ─────────────────────────────────────────────

def get_sentiment_weights(chat_msg: str = ""):
    """
    NOVELTY — Sentiment-adaptive fusion weights.

    Keyword analysis of an optional chat message shifts the
    balance between NCF (α), text (β), and poster (γ) signals:

      sad/stressed  → text-heavy  (α=0.50, β=0.35, γ=0.15)
      happy/excited → poster-heavy (α=0.60, β=0.20, γ=0.20)
      neutral       → default     (α=0.60, β=0.25, γ=0.15)

    Returns (alpha, beta, gamma).
    """
    msg = chat_msg.lower()
    sad_keywords    = {"sad", "bad", "angry", "stressed", "frustrated",
                       "depressed", "lonely", "anxious", "tired", "bored"}
    happy_keywords  = {"happy", "good", "fun", "excited", "great",
                       "amazing", "wonderful", "joy", "love", "cheerful"}

    if any(w in msg for w in sad_keywords):
        return 0.50, 0.35, 0.15   # lean on story/text for comfort
    elif any(w in msg for w in happy_keywords):
        return 0.60, 0.20, 0.20   # let poster visual mood shine
    else:
        return 0.60, 0.25, 0.15   # balanced default


def get_weight_explanation(chat_msg: str, alpha: float, beta: float, gamma: float) -> str:
    """Human-readable explanation of the chosen weight profile."""
    msg = chat_msg.lower()
    if any(w in msg for w in {"sad", "bad", "angry", "stressed", "frustrated"}):
        return "😢 **Sad/stressed mood** → text-heavy (stories for comfort)"
    elif any(w in msg for w in {"happy", "good", "fun", "excited", "great"}):
        return "😊 **Happy mood** → poster-boosted (visual energy)"
    else:
        return "➡️ **Neutral** → standard hybrid weights"


# ─────────────────────────────────────────────
# MAIN RECOMMENDER
# ─────────────────────────────────────────────

def recommend_movies_multimodal(
    user_id,
    context,
    chat_msg: str = "",
    top_n: int = 10,
):
    """
    Hybrid multimodal recommender combining:

      1. Neural Collaborative Filtering (NCF)        — who liked what
      2. Time-decay SBERT text embeddings            — narrative similarity
      3. Genre-conditioned poster embeddings         — visual + genre match
      4. Sentiment-dynamic fusion weights            — mood-aware balance

    Parameters
    ----------
    user_id   : int   — existing user ID from ratings
    context   : dict  — loaded resources from load_all()
    chat_msg  : str   — optional mood/chat input for dynamic weights
    top_n     : int   — number of recommendations to return

    Returns
    -------
    recommendations : pd.DataFrame  — top-N movies
    viz_data        : pd.DataFrame  — per-candidate score breakdown
    """
    movies           = context["movies"]
    ratings          = context["ratings"]
    ncf_model        = context["ncf_model"]
    user_encoder     = context["user_encoder"]
    movie_encoder    = context["movie_encoder"]
    text_embeddings  = context["text_embeddings"]
    poster_embeddings = context["poster_embeddings"]

    movie_id_to_index = dict(zip(movies["movieId"], movies.index))

    # ── 1. Sentiment-dynamic weights ──────────────────────────────
    alpha, beta, gamma = get_sentiment_weights(chat_msg)

    # ── 2. Candidate generation ───────────────────────────────────
    user_encoded = user_encoder.transform([user_id])[0]
    seen_movies  = set(ratings[ratings.userId == user_id]["movieId"])

    candidate_movies  = []
    candidate_indices = []

    for mid in movies["movieId"]:
        if mid in seen_movies:
            continue
        if mid not in movie_encoder.classes_:
            continue
        idx = movie_id_to_index.get(mid)
        if idx is None or idx >= len(text_embeddings):
            continue
        candidate_movies.append(mid)
        candidate_indices.append(idx)

    if not candidate_movies:
        empty = movies.sample(top_n)[["movieId", "title", "genres", "poster_path"]]
        return empty, pd.DataFrame()

    candidate_encoded = movie_encoder.transform(candidate_movies)

    # ── 3. NCF scores ─────────────────────────────────────────────
    user_input  = np.full(len(candidate_encoded), user_encoded)
    ncf_raw     = ncf_model.predict([user_input, candidate_encoded], verbose=0).flatten()
    ncf_scores  = normalize(ncf_raw)

    # ── 4. Text similarity (time-decay profile) ───────────────────
    user_text_vec = build_user_vector_time_decay(
        user_id, ratings, text_embeddings, movie_id_to_index
    )
    if user_text_vec is not None:
        text_scores = normalize(
            cosine_similarity(
                user_text_vec.reshape(1, -1),
                text_embeddings[candidate_indices],
            )[0]
        )
    else:
        text_scores = np.zeros(len(candidate_movies))

    # ── 5. Poster similarity (genre-conditioned) ──────────────────
    user_genre_profile = get_user_genre_profile(user_id, ratings, movies)
    user_poster_vec    = build_user_vector_time_decay(
        user_id, ratings, poster_embeddings, movie_id_to_index
    )

    if user_poster_vec is not None:
        poster_scores = genre_conditioned_poster_scores(
            candidate_movies  = candidate_movies,
            movies            = movies,
            poster_embeddings = poster_embeddings,
            user_poster_vec   = user_poster_vec,
            candidate_indices = candidate_indices,
            user_genre_profile= user_genre_profile,
        )
    else:
        poster_scores = np.zeros(len(candidate_movies))

    # ── 6. Hybrid fusion ──────────────────────────────────────────
    final_scores = (
        alpha * ncf_scores +
        beta  * text_scores +
        gamma * poster_scores
    )

    top_idx  = np.argsort(final_scores)[::-1][:top_n]
    rec_ids  = [candidate_movies[i] for i in top_idx]

    recommendations = movies[movies.movieId.isin(rec_ids)][
        ["movieId", "title", "genres", "poster_path"]
    ]

    # ── 7. Visualisation payload ──────────────────────────────────
    weight_tag = f"α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}"
    viz_data = pd.DataFrame({
        "movieId" : [candidate_movies[i] for i in top_idx],
        "title"   : [
            movies[movies.movieId == candidate_movies[i]]["title"].values[0]
            for i in top_idx
        ],
        "ncf"     : ncf_scores[top_idx],
        "text"    : text_scores[top_idx],
        "poster"  : poster_scores[top_idx],
        "final"   : final_scores[top_idx],
        "weights" : weight_tag,
    })

    return recommendations, viz_data