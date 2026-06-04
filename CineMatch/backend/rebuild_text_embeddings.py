# backend/rebuild_text_embeddings.py
"""
Run this ONCE to rebuild text_embeddings.npy with rich combined text.
Rich text = title + genres + overview + tagline + keywords

This fixes the core semantic search problem:
  BEFORE: "Zootopia" embedding had only "Zootopia Animation Comedy Crime"
  AFTER:  "Zootopia" embedding has plot, animals, fox, rabbit, police etc.

Usage:
    cd D:\MINI PROJECT
    python backend/rebuild_text_embeddings.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from backend.config import MOVIES_PATH, TEXT_EMB_PATH

def build_combined_text(row) -> str:
    title    = str(row.get("title", "")    or "")
    genres   = str(row.get("genres", "")   or "").replace("|", " ")
    overview = str(row.get("overview", "") or "")[:400]   # cap at 400 chars
    tagline  = str(row.get("tagline", "")  or "")
    keywords = str(row.get("keywords", "") or "")[:200]

    # Title repeated twice — gives it more weight in the embedding
    combined = f"{title} {title} {genres} {overview} {tagline} {keywords}"
    return " ".join(combined.split())

def main():
    print("Loading movies...")
    movies = pd.read_csv(MOVIES_PATH)
    print(f"  {len(movies)} movies loaded")

    # Check which columns are available
    available = movies.columns.tolist()
    print(f"  Columns: {available}")

    has_overview = "overview" in available and movies["overview"].notna().sum() > 100
    print(f"  Has overview: {has_overview} "
          f"({movies['overview'].notna().sum() if has_overview else 0} non-null)")

    print("Building combined text...")
    movies["_combined_text"] = movies.apply(build_combined_text, axis=1)

    # Show a sample
    print("\nSample combined text (first movie):")
    print(" ", movies["_combined_text"].iloc[0][:200])

    print("\nEncoding with SBERT (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    embeddings = model.encode(
        movies["_combined_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=128,
    )
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Save — overwrites existing TEXT_EMB_PATH
    import os
    backup_path = TEXT_EMB_PATH.replace(".npy", "_backup_old.npy")
    if os.path.exists(TEXT_EMB_PATH):
        old = np.load(TEXT_EMB_PATH)
        np.save(backup_path, old)
        print(f"Old embeddings backed up to: {backup_path}")

    np.save(TEXT_EMB_PATH, embeddings)
    print(f"New embeddings saved to: {TEXT_EMB_PATH}")
    print("\nDone! Restart Streamlit to use the new embeddings.")

if __name__ == "__main__":
    main()