# ================================
# STEP 1: IMPORTS
# ================================
import pandas as pd
import numpy as np
import sys, os

# Adjust path if needed
PROJECT_ROOT = "D:/MINI PROJECT"
sys.path.append(PROJECT_ROOT)

# ================================
# STEP 2: LOAD CONTEXT
# ================================
from backend.loaders import load_all

context = load_all()

# ================================
# STEP 3: ACCESS DATA
# ================================
movies = context["movies"]
ratings = context["ratings"]
embeddings = context["text_embeddings"]

# ================================
# STEP 4: BASIC CHECKS
# ================================
print("Total movies:", len(movies))
print("Unique movieIds:", movies["movieId"].nunique())
print("Embeddings shape:", embeddings.shape)

# ================================
# STEP 5: FIND DUPLICATES
# ================================
duplicates = movies[movies.duplicated(subset="movieId", keep=False)]

print("\nDuplicate rows count:", len(duplicates))

if len(duplicates) > 0:
    print("\nSample duplicates:")
    print(duplicates[["movieId", "title"]].head(10))

# ================================
# STEP 6: CHECK YEAR COLUMN
# ================================
print("\nColumns:", movies.columns)

if "release_year" in movies.columns:
    print("Max year:", movies["release_year"].max())
else:
    print("release_year NOT FOUND")

# ================================
# STEP 7: FINAL CONSISTENCY CHECK
# ================================
if len(movies) != embeddings.shape[0]:
    print("\n❌ PROBLEM: Movies and embeddings mismatch!")
else:
    print("\n✅ Movies and embeddings aligned")