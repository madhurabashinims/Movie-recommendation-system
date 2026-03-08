# backend/predictors.py

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
# backend/predictors.py

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Poster embedding extractor (ResNet)
# -------------------------------------------------
def extract_poster_embedding(image, poster_model, device="cpu"):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = poster_model(image_tensor)

    return features.squeeze().cpu().numpy()


def predict_user_like_movie(
    user_id,
    movie_text,
    context,
    uploaded_poster=None,
    alpha=0.7,
    beta=0.3
):
    ratings = context["ratings"]
    movies = context["movies"]
    text_embeddings = context["text_embeddings"]
    poster_embeddings = context["poster_embeddings"]
    text_model = context["text_model"]
    poster_model = context["poster_model"]

    movie_id_to_index = dict(zip(movies.movieId, movies.index))

    # ---- USER HISTORY ----
    seen_movie_ids = ratings[ratings.userId == user_id]["movieId"]
    indices = [movie_id_to_index[mid] for mid in seen_movie_ids if mid in movie_id_to_index]

    if not indices:
        return {"final_score": 0.0, "verdict": "unknown", "confidence": "low"}

    # ---- TEXT SIMILARITY ----
    user_text_vec = text_embeddings[indices].mean(axis=0)
    movie_text_vec = text_model.encode([movie_text])[0]

    text_score = cosine_similarity(
        user_text_vec.reshape(1, -1),
        movie_text_vec.reshape(1, -1)
    )[0][0]

    # ---- POSTER SIMILARITY (OPTIONAL) ----
    poster_score = 0.0
    used_poster = False

    if uploaded_poster is not None:
        img = Image.open(uploaded_poster).convert("RGB")
        poster_vec = extract_poster_embedding(img, poster_model)

        user_poster_vec = poster_embeddings[indices]
        user_poster_vec = user_poster_vec[np.linalg.norm(user_poster_vec, axis=1) > 0]

        if len(user_poster_vec) > 0:
            user_poster_vec = user_poster_vec.mean(axis=0)
            poster_score = cosine_similarity(
                user_poster_vec.reshape(1, -1),
                poster_vec.reshape(1, -1)
            )[0][0]
            used_poster = True

    # ---- FINAL SCORE ----
    final_score = alpha * text_score + beta * poster_score

    # ---- VERDICT ----
    if final_score >= 0.6:
        verdict, confidence = "like", "high"
    elif final_score >= 0.4:
        verdict, confidence = "maybe", "medium"
    else:
        verdict, confidence = "unlikely", "low"

    return {
        "final_score": float(final_score),
        "verdict": verdict,
        "confidence": confidence,
        "text_score": float(text_score),
        "poster_score": float(poster_score),
        "used_poster": used_poster
    }
