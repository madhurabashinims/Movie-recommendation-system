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
    POSTER_EMB_PATH
)


def load_all():
   
    # ---------------- DATA ----------------
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    train_ratings, test_ratings = split_ratings(ratings)
    # ---------------- MODELS ----------------
    ncf_model = load_model(NCF_MODEL_PATH)
    user_encoder = joblib.load(USER_ENCODER_PATH)
    movie_encoder = joblib.load(MOVIE_ENCODER_PATH)

    # ---------------- EMBEDDINGS ----------------
    text_embeddings = np.load(TEXT_EMB_PATH)
    poster_embeddings = np.load(POSTER_EMB_PATH)

    # ---------------- TEXT MODEL ----------------
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------- POSTER MODEL (NEW) ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    poster_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    poster_model.fc = torch.nn.Identity()
    poster_model.eval().to(device)

    poster_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ---- Poster model (ResNet-50) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    poster_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    poster_model.fc = torch.nn.Identity()
    poster_model = poster_model.to(device)
    poster_model.eval()

    print("Train ratings:", len(train_ratings))
    print("Test ratings:", len(test_ratings))
    return {
    "movies": movies,
    "ratings": train_ratings,
    "test_ratings": test_ratings,
    "ncf_model": ncf_model,
    "user_encoder": user_encoder,
    "movie_encoder": movie_encoder,
    "text_embeddings": text_embeddings,
    "poster_embeddings": poster_embeddings,
    "text_model": text_model,
    "poster_model": poster_model,
    "poster_preprocess": poster_preprocess,
    "device": device
}
