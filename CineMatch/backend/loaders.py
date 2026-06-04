# backend/loaders.py

import joblib
import numpy as np
import pandas as pd
import torch
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms

from backend.train_test_split import split_ratings
from backend.config import (
    NCF_MODEL_PATH,
    USER_ENCODER_PATH,
    MOVIE_ENCODER_PATH,
    MOVIES_PATH,
    RATINGS_PATH,
    TEXT_EMB_PATH,
    POSTER_EMB_PATH,
)


def load_all():

    # ── Data ──────────────────────────────────────────────────────
    movies  = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    train_ratings, test_ratings = split_ratings(ratings)

    # ── NCF model + encoders ──────────────────────────────────────
    ncf_model     = load_model(NCF_MODEL_PATH)
    user_encoder  = joblib.load(USER_ENCODER_PATH)
    movie_encoder = joblib.load(MOVIE_ENCODER_PATH)

    # ── Precomputed embeddings ────────────────────────────────────
    text_embeddings   = np.load(TEXT_EMB_PATH)
    poster_embeddings = np.load(POSTER_EMB_PATH)   # ← must be poster_embeddings.npy (9k)

    # ── SBERT text model ──────────────────────────────────────────
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── ResNet-50 poster model (loaded ONCE) ──────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    poster_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    poster_model.fc = torch.nn.Identity()   # strip classifier → 2048-dim features
    poster_model.eval().to(device)

    poster_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print(f"✅ Loaded {len(movies)} movies, {len(ratings)} ratings")
    print(f"✅ Text embeddings : {text_embeddings.shape}")
    print(f"✅ Poster embeddings: {poster_embeddings.shape}")
    print(f"✅ Non-zero posters : {int((np.linalg.norm(poster_embeddings, axis=1) > 0).sum())}")
    print(f"✅ Device           : {device}")
    print(f"   Train ratings   : {len(train_ratings)}")
    print(f"   Test ratings    : {len(test_ratings)}")

    return {
        "movies"            : movies,
        "ratings"           : train_ratings,
        "test_ratings"      : test_ratings,
        "ncf_model"         : ncf_model,
        "user_encoder"      : user_encoder,
        "movie_encoder"     : movie_encoder,
        "text_embeddings"   : text_embeddings,
        "poster_embeddings" : poster_embeddings,
        "text_model"        : text_model,
        "poster_model"      : poster_model,
        "poster_preprocess" : poster_preprocess,
        "device"            : device,
    }