# ===============================
# Hybrid Recommendation System
# ===============================

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PATHS (EDIT ONLY IF NEEDED)
# -------------------------------
NCF_MODEL_PATH = r"D:\MINI PROJECT\Checkpoints\ncf_model.keras"
USER_ENCODER_PATH = r"D:\MINI PROJECT\Checkpoints\user_encoder.pkl"
MOVIE_ENCODER_PATH = r"D:\MINI PROJECT\Checkpoints\movie_encoder.pkl"
EMBEDDINGS_PATH = r"D:\MINI PROJECT\Checkpoints\movie_embeddings.npy"

MOVIES_PATH = r"D:\MINI PROJECT\DATASET\movies_final.csv"
RATINGS_PATH = r"D:\MINI PROJECT\DATASET\MovieLensDataset20M\rating.csv"

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading data...")
movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

# -------------------------------
# LOAD MODEL & ENCODERS
# -------------------------------
print("Loading NCF model...")
ncfmodel = load_model(NCF_MODEL_PATH)

print("Loading encoders...")
user_encoder = joblib.load(USER_ENCODER_PATH)
movie_encoder = joblib.load(MOVIE_ENCODER_PATH)

# -------------------------------
# LOAD OR CREATE EMBEDDINGS
# -------------------------------
try:
    embeddings = np.load(EMBEDDINGS_PATH)
    print("Loaded existing embeddings:", embeddings.shape)

    # sanity check
    if embeddings.shape[0] != len(movies):
        raise ValueError("Embedding count mismatch")

except Exception as e:
    print("Rebuilding embeddings due to error:", e)

    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    movies["content_text"] = (
        movies["title"].fillna("") + " " +
        movies["genres"].fillna("") + " " +
        movies["overview"].fillna("")
    )

    embeddings = text_model.encode(
        movies["content_text"].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    print("Saved embeddings:", embeddings.shape)


# -------------------------------
# HYBRID RECOMMENDER
# -------------------------------
def recommend_movies_hybrid(
    user_id,
    ncf_model,
    ratings_df,
    movies_df,
    embeddings,
    user_encoder,
    movie_encoder,
    alpha=0.7,
    top_n=10
):
    user_encoded = user_encoder.transform([user_id])[0]

    seen_movies = set(
        ratings_df[ratings_df.userId == user_id]["movieId"]
    )

    candidate_movies = [
        m for m in movies_df["movieId"]
        if m not in seen_movies and m in movie_encoder.classes_
    ]

    candidate_encoded = movie_encoder.transform(candidate_movies)

    # NCF scores
    user_input = np.full(len(candidate_encoded), user_encoded)
    ncf_scores = ncf_model.predict(
        [user_input, candidate_encoded],
        batch_size=1024,
        verbose=0
    ).flatten()

    ncf_scores = (ncf_scores - ncf_scores.min()) / (np.ptp(ncf_scores) + 1e-8)

    # Content scores
    id_to_index = dict(zip(movies_df.movieId, movies_df.index))
    liked_indices = [
        id_to_index[mid] for mid in seen_movies if mid in id_to_index
    ]

    user_vector = embeddings[liked_indices].mean(axis=0)
    candidate_indices = [id_to_index[mid] for mid in candidate_movies]

    content_scores = cosine_similarity(
        user_vector.reshape(1, -1),
        embeddings[candidate_indices]
    )[0]

    content_scores = (content_scores - content_scores.min()) / (np.ptp(content_scores) + 1e-8)


    # Hybrid score
    final_scores = alpha * ncf_scores + (1 - alpha) * content_scores

    top_idx = np.argsort(final_scores)[::-1][:top_n]
    rec_ids = [candidate_movies[i] for i in top_idx]

    return movies_df[movies_df.movieId.isin(rec_ids)][
        ["movieId", "title", "genres"]
    ]

# -------------------------------
# TEST
# -------------------------------
print("\nGenerating hybrid recommendations...\n")
recs = recommend_movies_hybrid(
    user_id=10,
    ncf_model=ncfmodel,
    ratings_df=ratings,
    movies_df=movies,
    embeddings=embeddings,
    user_encoder=user_encoder,
    movie_encoder=movie_encoder,
    alpha=0.7,
    top_n=10
)

print(recs)
