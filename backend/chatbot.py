# backend/chatbot.py
"""
Emotion-aware chatbot recommendation backend.
Uses context detection + semantic scoring + recency boosting.
"""

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── lazy model ────────────────────────────────────────────────
_query_model = None
def get_query_model():
    global _query_model
    if _query_model is None:
        _query_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _query_model


# ══════════════════════════════════════════════════════════════
#  CONTEXT / EMOTION DETECTION
# ══════════════════════════════════════════════════════════════
CONTEXT_RULES = [
    # (context_label, keywords)
    ("deep_grief",    ["mom", "mother", "dad", "father", "parents", "grandma",
                       "grandpa", "passed away", "died", "funeral", "loss",
                       "miss them", "missing them"]),
    ("soft_grief",    ["pet", "dog", "cat", "bunny", "rabbit", "hamster",
                       "fish died", "lost my pet"]),
    ("romantic_loss", ["breakup", "broke up", "ex", "heartbreak", "dumped",
                       "she left", "he left", "divorce"]),
    ("anger",         ["angry", "mad", "furious", "yelled", "yell", "boss",
                       "frustrated", "annoyed", "pissed", "argument",
                       "fight", "argument", "unfair", "rant"]),
    ("stressed",      ["stressed", "stress", "overwhelmed", "anxious",
                       "anxiety", "panic", "deadline", "pressure",
                       "burned out", "exhausted", "tired"]),
    ("sad",           ["sad", "depressed", "lonely", "alone", "cry",
                       "crying", "unhappy", "hopeless", "empty",
                       "miss", "missing"]),
    ("happy",         ["happy", "great day", "amazing", "excited",
                       "wonderful", "fantastic", "celebrate",
                       "friend", "good day", "best day", "joy"]),
    ("bored",         ["bored", "nothing to do", "kill time",
                       "free time", "lazy day", "chill"]),
]

def detect_context(message: str) -> str:
    msg = message.lower()
    for ctx, keywords in CONTEXT_RULES:
        if any(k in msg for k in keywords):
            return ctx
    return "general"

def get_target_genres(context: str) -> list:
    mapping = {
        "deep_grief":    ["Drama", "Family"],
        "soft_grief":    ["Animation", "Family", "Adventure"],
        "romantic_loss": ["Romance", "Drama", "Comedy"],
        "anger":         ["Action", "Comedy", "Thriller"],
        "stressed":      ["Comedy", "Animation", "Family"],
        "sad":           ["Drama", "Romance", "Animation"],
        "happy":         ["Comedy", "Animation", "Adventure", "Family"],
        "bored":         ["Action", "Adventure", "Thriller", "Comedy"],
        "general":       ["Comedy", "Family", "Animation"],
    }
    return mapping.get(context, ["Comedy", "Drama"])

def format_mood(context: str) -> str:
    labels = {
        "deep_grief":    "Deep emotion",
        "soft_grief":    "Tough day",
        "romantic_loss": "Heartbreak",
        "anger":         "Feeling frustrated",
        "stressed":      "Stressed out",
        "sad":           "Feeling low",
        "happy":         "Feeling good",
        "bored":         "Looking for fun",
        "general":       "Just browsing",
    }
    return labels.get(context, "Movie mood")

def generate_reply(message: str, context: str) -> str:
    replies = {
        "deep_grief":    ("I'm so sorry for your loss. It can be really hard to lose someone "
                          "so important. Here are some meaningful films that might bring comfort."),
        "soft_grief":    ("Losing a little companion leaves such a quiet emptiness. "
                          "Here are some warm, comforting films."),
        "romantic_loss": ("Heartbreak is never easy. Maybe a good film will help you "
                          "process your feelings."),
        "anger":         ("I'm sorry you're dealing with that — it sounds really frustrating. "
                          "Here are some films that might help you unwind."),
        "stressed":      ("Sounds like a tough time. Let's find something that helps you "
                          "switch off and relax."),
        "sad":           ("I'm sorry you're feeling this way. Here are some films "
                          "that might help lift your spirits."),
        "happy":         ("Love that energy! Here are some films to keep the good mood going."),
        "bored":         ("Let's fix that — here are some genuinely entertaining picks."),
        "general":       ("Here are some films you might enjoy."),
    }
    # Try Groq for personalised reply
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": (
                f"User said: \"{message}\". Detected mood: {context}. "
                f"Write a warm 1-2 sentence empathetic reply acknowledging their feeling "
                f"and mentioning you found movies for them. Keep it natural and brief. "
                f"No emojis."
            )}],
            model="llama-3.1-8b-instant",
            max_tokens=80,
        )
        reply = completion.choices[0].message.content.strip()
        if reply:
            return reply
    except Exception:
        pass
    return replies.get(context, replies["general"])


# ══════════════════════════════════════════════════════════════
#  SCORING HELPERS
# ══════════════════════════════════════════════════════════════
def genre_score(genres_str: str, targets: list) -> float:
    if pd.isna(genres_str) or not genres_str:
        return 0.0
    g = genres_str.lower()
    hits = sum(1 for t in targets if t.lower() in g)
    return hits / max(len(targets), 1)

def extract_keywords(message: str) -> list:
    patterns = [
        r"\b(bunny|rabbit|dog|cat|animal|animals|pet|zoo|lion|tiger|bear|fox|wolf|bird|fish|penguin|panda)\b",
        r"\b(adventure|magic|family|friendship|love|robot|space|war|ocean|jungle|school)\b",
        r"\b(animation|animated|cartoon|pixar|disney|anime)\b",
        r"\b(funny|happy|dark|emotional|thrilling|scary|uplifting|cheerful|wholesome|motivational)\b",
        r"\b(heist|detective|spy|superhero|musical|historical|western|mystery|sci-fi|fantasy|horror)\b",
    ]
    kws = []
    for p in patterns:
        kws.extend(re.findall(p, message.lower()))
    return list(set(kws))

def build_semantic_query(message: str, context: str, genres: list) -> str:
    kws = extract_keywords(message)
    parts = [message]
    if kws:
        parts.append(" ".join(kws))
    if genres:
        parts.append(" ".join(genres))
    parts.append(context)
    return " ".join(parts)

def keyword_boost(df: pd.DataFrame, message: str) -> pd.Series:
    kws = extract_keywords(message)
    if not kws:
        return pd.Series(0.0, index=df.index)
    pattern = "|".join(kws)
    title   = df["title"].str.contains(pattern, case=False, na=False).astype(float)
    genres  = df["genres"].str.contains(pattern, case=False, na=False).astype(float)
    overview = pd.Series(0.0, index=df.index)
    if "overview" in df.columns:
        overview = df["overview"].str.contains(pattern, case=False, na=False).astype(float)
    return 0.30 * title + 0.25 * overview + 0.20 * genres


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def chatbot_recommendation(user_id, message: str, context: dict) -> dict:
    """
    Returns dict with keys:
      reply, recommendations, mood_display, genres, confidence
    """
    model  = get_query_model()

    # ── Safe dedup ────────────────────────────────────────────
    movies = (context["movies"]
              .drop_duplicates(subset="movieId")
              .reset_index(drop=True))
    embeddings = context["text_embeddings"]
    ratings    = context["ratings"]

    # movieId → positional embedding row index
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies["movieId"])}

    # ── Detect context / emotion ──────────────────────────────
    ctx    = detect_context(message)
    genres = get_target_genres(ctx)

    # ── Candidate pool ────────────────────────────────────────
    # Start with ALL movies (not NCF-filtered) for chatbot
    # to avoid cold-start for new / recent movies
    candidates = movies.copy()

    # Remove movies user has already rated
    seen_ids = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
    candidates = candidates[~candidates["movieId"].isin(seen_ids)].copy()
    candidates = candidates.reset_index(drop=True)

    # ── Genre score ───────────────────────────────────────────
    candidates["genre_score"] = candidates["genres"].apply(
        lambda g: genre_score(g, genres)
    )

    # ── Semantic score ────────────────────────────────────────
    query   = build_semantic_query(message, ctx, genres)
    q_vec   = model.encode([query])[0]
    all_sc  = cosine_similarity([q_vec], embeddings)[0]
    sc_map  = {mid: float(all_sc[i])
               for mid, i in movie_id_to_idx.items()
               if i < len(all_sc)}
    candidates["semantic_score"] = (
        candidates["movieId"].map(sc_map).fillna(0.0)
    )

    # ── Keyword boost ─────────────────────────────────────────
    candidates["semantic_score"] = (
        candidates["semantic_score"] + keyword_boost(candidates, message)
    ).clip(upper=1.0)

    # ── Animal / kids boost ───────────────────────────────────
    if any(x in message.lower() for x in
           ["animal","animals","bunny","rabbit","pet","zoo",
            "kids","children","toddler"]):
        mask = candidates["genres"].str.contains(
            "Animation|Family|Adventure", case=False, na=False)
        candidates.loc[mask, "semantic_score"] += 0.25

    # ── Rating boost ──────────────────────────────────────────
    if "vote_average" in candidates.columns:
        candidates["rating_score"] = (
            pd.to_numeric(candidates["vote_average"], errors="coerce")
            .fillna(5.0) / 10.0
        )
    else:
        candidates["rating_score"] = 0.5

    # ── Recency boost (plain dict — no Series.map) ────────────
    yr_map: dict = {}
    if "release_date" in movies.columns:
        yr_s   = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
        yr_map = dict(zip(movies["movieId"], yr_s))
    elif "release_year" in movies.columns:
        yr_map = dict(zip(movies["movieId"],
                          pd.to_numeric(movies["release_year"], errors="coerce")))

    def _year_boost(mid):
        y = yr_map.get(mid, np.nan)
        if pd.isna(y): return 0.0
        if y >= 2022:  return 0.18
        if y >= 2019:  return 0.12
        if y >= 2015:  return 0.07
        if y >= 2010:  return 0.03
        return 0.0

    candidates["recent_boost"] = candidates["movieId"].apply(_year_boost)

    # ── Popularity ────────────────────────────────────────────
    if "popularity" in candidates.columns:
        pop = pd.to_numeric(candidates["popularity"], errors="coerce").fillna(0)
        pop_max = pop.max()
        candidates["pop_score"] = pop / (pop_max + 1e-8) if pop_max > 0 else 0.3
    else:
        candidates["pop_score"] = 0.3

    # ── Final score ───────────────────────────────────────────
    candidates["final_score"] = (
        0.40 * candidates["semantic_score"] +
        0.25 * candidates["genre_score"] +
        0.15 * candidates["recent_boost"] +
        0.10 * candidates["rating_score"] +
        0.10 * candidates["pop_score"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Only keep movies that match at least one target genre
    genre_filtered = candidates[candidates["genre_score"] > 0]
    if len(genre_filtered) >= 6:
        candidates = genre_filtered

    # ── Diversity filter ──────────────────────────────────────
    sorted_cands = candidates.sort_values("final_score", ascending=False)
    recs, used_genre_sets = [], []
    for _, row in sorted_cands.iterrows():
        if len(recs) >= 10:
            break
        g_set = frozenset(
            g.strip().lower()
            for g in str(row.get("genres","")).split("|")
            if g.strip()
        )
        # Allow if genre overlap with existing selections < 80%
        too_similar = any(
            len(g_set & prev) / max(len(g_set | prev), 1) > 0.8
            for prev in used_genre_sets
        )
        if not too_similar or len(recs) < 4:
            recs.append(row)
            used_genre_sets.append(g_set)

    top = pd.DataFrame(recs).reset_index(drop=True)
    if len(top) < 6:
        top = sorted_cands.head(6).reset_index(drop=True)

    # ── Reasons ──────────────────────────────────────────────
    kws = extract_keywords(message)
    def build_reason(row):
        parts      = []
        genres_str = str(row.get("genres", ""))
        if kws and re.search("|".join(kws), genres_str, re.IGNORECASE):
            parts.append("Highly similar to your message")
        if genres_str:
            g_list = [g.strip() for g in genres_str.split("|") if g.strip()][:3]
            parts.append("Genre: " + ", ".join(g_list).lower())
        parts.append(f"Mood alignment: {ctx}")
        return " · ".join(parts) if parts else "Matches your mood and interests"

    top["reason"] = top.apply(build_reason, axis=1)

    # Ensure poster_path exists
    if "poster_path" not in top.columns:
        top["poster_path"] = None

    out_cols = [c for c in
                ["movieId","title","genres","poster_path","reason","final_score"]
                if c in top.columns]

    return {
        "reply":           generate_reply(message, ctx),
        "mood_display":    format_mood(ctx),
        "genres":          genres,
        "confidence":      0.85,
        "recommendations": top[out_cols].rename(columns={"final_score": "score"}),
    }