# backend/ultra_filter.py
"""
Ultra-Specific Movie Recommendation Pipeline — 3-Stage
Stage 1: Hard genre/era filters
Stage 2: Semantic search (runtime embeddings)  
Stage 3: Attribute boosting + novelty modifiers

Exports: ultra_filter_recommend, generate_ultra_explanation, GENRE_ALIASES
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Lazy model ────────────────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


# ══════════════════════════════════════════════════════════════
#  EXPORTED CONSTANTS
# ══════════════════════════════════════════════════════════════
GENRE_ALIASES = {
    "action":          "Action",
    "adventure":       "Adventure",
    "animation":       "Animation",
    "comedy":          "Comedy",
    "crime":           "Crime",
    "documentary":     "Documentary",
    "drama":           "Drama",
    "fantasy":         "Fantasy",
    "family":          "Family",
    "history":         "History",
    "horror":          "Horror",
    "music":           "Music",
    "mystery":         "Mystery",
    "romance":         "Romance",
    "sci-fi":          "Science Fiction",
    "science fiction": "Science Fiction",
    "thriller":        "Thriller",
    "war":             "War",
    "western":         "Western",
}

ATTRIBUTE_BOOSTS = {
    "animal":       (["Animation","Adventure","Family"],       0.14),
    "animals":      (["Animation","Adventure","Family"],       0.14),
    "bunny":        (["Animation","Family"],                   0.16),
    "rabbit":       (["Animation","Family"],                   0.16),
    "dog":          (["Animation","Family","Comedy"],          0.10),
    "cat":          (["Animation","Family","Comedy"],          0.10),
    "pet":          (["Animation","Family","Comedy"],          0.10),
    "kids":         (["Animation","Family","Comedy"],          0.14),
    "children":     (["Animation","Family"],                   0.12),
    "family":       (["Family","Animation","Comedy"],          0.10),
    "animation":    (["Animation"],                            0.18),
    "animated":     (["Animation"],                            0.18),
    "cartoon":      (["Animation"],                            0.16),
    "anime":        (["Animation"],                            0.14),
    "comedy":       (["Comedy"],                               0.12),
    "horror":       (["Horror"],                               0.14),
    "thriller":     (["Thriller","Mystery"],                   0.12),
    "romance":      (["Romance"],                              0.12),
    "romantic":     (["Romance"],                              0.10),
    "action":       (["Action"],                               0.12),
    "adventure":    (["Adventure","Action","Family"],          0.12),
    "adventurous":  (["Adventure","Action","Family"],          0.12),
    "fantasy":      (["Fantasy"],                              0.12),
    "documentary":  (["Documentary"],                          0.14),
    "mystery":      (["Mystery","Thriller"],                   0.12),
    "western":      (["Western"],                              0.14),
    "superhero":    (["Action","Science Fiction"],             0.14),
    "sci-fi":       (["Science Fiction"],                      0.14),
    "scifi":        (["Science Fiction"],                      0.14),
    "space":        (["Science Fiction","Adventure"],          0.12),
    "war":          (["War","Drama","Action"],                 0.12),
    "heist":        (["Crime","Thriller","Comedy"],            0.12),
    "cheerful":     (["Comedy","Animation","Family"],          0.12),
    "uplifting":    (["Family","Animation","Comedy"],          0.12),
    "wholesome":    (["Family","Animation","Comedy"],          0.12),
    "heartwarming": (["Family","Drama","Romance"],             0.12),
    "funny":        (["Comedy"],                               0.12),
    "lighthearted": (["Comedy","Family","Animation"],          0.12),
    "fun":          (["Comedy","Animation","Family"],          0.10),
    "motivational": (["Drama","Family","Adventure","Action"],  0.12),
    "inspiring":    (["Drama","Family","Adventure"],           0.12),
    "dark":         (["Thriller","Crime","Drama"],             0.10),
    "scary":        (["Horror","Thriller"],                    0.12),
    "intense":      (["Thriller","Action","Drama"],            0.10),
    "mind-bending": (["Science Fiction","Thriller","Mystery"], 0.14),
    "psychological":(["Thriller","Drama","Mystery"],           0.12),
    "emotional":    (["Drama","Romance","Family"],             0.10),
    "nostalgic":    (["Drama","Family","Animation"],           0.10),
    "magical":      (["Fantasy","Animation","Family"],         0.12),
    "epic":         (["Action","Adventure","Fantasy","Drama"], 0.10),
    "thrilling":    (["Thriller","Action","Adventure"],        0.12),
    "ocean":        (["Adventure","Animation","Family"],       0.10),
    "jungle":       (["Adventure","Animation","Family"],       0.10),
    "princess":     (["Animation","Fantasy","Family"],         0.12),
    "dragon":       (["Fantasy","Animation","Adventure"],      0.12),
    "robot":        (["Science Fiction","Animation"],          0.12),
    "detective":    (["Mystery","Crime","Thriller"],           0.12),
}

INCLUSION_KEYWORDS = {
    "animation":   "Animation",
    "animated":    "Animation",
    "cartoon":     "Animation",
    "anime":       "Animation",
    "kids":        "Family|Animation",
    "children":    "Family|Animation",
    "family":      "Family",
    "comedy":      "Comedy",
    "horror":      "Horror",
    "thriller":    "Thriller",
    "action":      "Action",
    "adventure":   "Adventure",
    "adventurous": "Adventure",
    "romance":     "Romance",
    "romantic":    "Romance",
    "documentary": "Documentary",
    "sci-fi":      "Science Fiction",
    "scifi":       "Science Fiction",
    "fantasy":     "Fantasy",
    "mystery":     "Mystery",
    "crime":       "Crime",
    "western":     "Western",
    "musical":     "Music|Musical",
    "superhero":   "Action|Science Fiction",
    "war":         "War",
    "heist":       "Crime|Thriller",
    "magical":     "Fantasy|Animation",
    "whimsical":   "Fantasy|Animation",
}

NEGATIVE_CONTENT = {
    "violent":    ["Horror"],
    "violence":   ["Horror"],
    "gore":       ["Horror"],
    "gory":       ["Horror"],
    "disturbing": ["Horror"],
}

ERA_MAP = {
    "1950s":(1950,1959), "50s":(1950,1959),
    "1960s":(1960,1969), "60s":(1960,1969),
    "1970s":(1970,1979), "70s":(1970,1979),
    "1980s":(1980,1989), "80s":(1980,1989),
    "1990s":(1990,1999), "90s":(1990,1999),
    "2000s":(2000,2009), "00s":(2000,2009),
    "2010s":(2010,2019), "10s":(2010,2019),
    "2020s":(2020,2029), "20s":(2020,2029),
    "retro":(1950,1995), "vintage":(1950,1990),
    "classic":(1950,2000), "modern":(2010,2029),
    "recent":(2018,2029), "new":(2020,2029),
    "latest":(2022,2029),
}

DELTA_MODIFIERS = {
    "simpler":   "simple accessible straightforward",
    "funnier":   "funny comedy humor lighthearted",
    "darker":    "dark gritty intense serious",
    "scarier":   "scary terrifying horror suspense",
    "lighter":   "light fun cheerful uplifting",
    "deeper":    "philosophical deep complex",
    "faster":    "fast-paced action energetic thrilling",
    "slower":    "slow-burn deliberate atmospheric",
    "romantic":  "romantic love relationship heartfelt",
    "epic":      "epic grand large-scale adventure",
    "realistic": "realistic grounded authentic",
}

NEGATION_PHRASES = [
    r"not like (.+?)(?:,|\.|$)",
    r"nothing like (.+?)(?:,|\.|$)",
    r"avoid (.+?)(?:,|\.|$)",
    r"without (.+?)(?:,|\.|$)",
    r"NOT like (.+?)(?:,|\.|$)",
    r"no (.+?) elements?",
    r"except (.+?)(?:,|\.|$)",
    r"not (.+?) (?:movies?|films?|vibes?)",
]


# ══════════════════════════════════════════════════════════════
#  PARSING
# ══════════════════════════════════════════════════════════════
def parse_query(query: str, movies_df: pd.DataFrame) -> dict:
    q       = query.strip()
    q_lower = q.lower()
    result  = {
        "free_text":      q,
        "negations":      [],
        "anchor_movie":   None,
        "anchor_delta":   [],
        "era_range":      None,
        "exclude_genres": [],
        "mood_text":      "",
        "theme_text":     "",
    }

    # Negations
    for pattern in NEGATION_PHRASES:
        for m in re.finditer(pattern, q, re.IGNORECASE):
            neg = m.group(1).strip().rstrip(".,")
            if neg and len(neg) > 1:
                result["negations"].append(neg)

    # Anchor movie
    anchor_pat = re.search(
        r"(?:like|similar to|inspired by)\s+([A-Z][^,\.]+?)(?:\s+but|\s+except|,|$)",
        q, re.IGNORECASE,
    )
    if anchor_pat:
        candidate = anchor_pat.group(1).strip()
        matched   = _fuzzy_match(candidate, movies_df)
        if matched:
            result["anchor_movie"] = matched
            but_m = re.search(r"but (.+?)(?:,|$)", q, re.IGNORECASE)
            if but_m:
                delta_text = but_m.group(1).lower()
                result["anchor_delta"] = [
                    DELTA_MODIFIERS[k]
                    for k in DELTA_MODIFIERS if k in delta_text
                ]

    # Era
    for era_kw, yr in ERA_MAP.items():
        if re.search(r"\b" + re.escape(era_kw) + r"\b", q_lower):
            result["era_range"] = yr
            break

    # Exclude genres — catches "no X", "not X", "without X", "avoid X", "exclude X"
    excl = re.findall(r"(?:no|without|exclude|not|avoid)\s+(\w+)", q_lower)
    for word in excl:
        if word in GENRE_ALIASES:
            result["exclude_genres"].append(GENRE_ALIASES[word])
        # Also catch direct genre names after negation words
        for alias_key, alias_val in GENRE_ALIASES.items():
            if word == alias_key.replace("-","").replace(" ",""):
                result["exclude_genres"].append(alias_val)
    result["exclude_genres"] = list(set(result["exclude_genres"]))

    # Mood / theme decoupling
    mood_words  = ["dark","hopeful","cheerful","emotional","intense","uplifting",
                   "sad","happy","scary","romantic","funny","gritty","dreamy"]
    theme_words = ["heist","space","ocean","war","school","love","robot","magic",
                   "adventure","mystery","family","sport","music","detective"]
    found_mood  = [w for w in mood_words  if w in q_lower]
    found_theme = [w for w in theme_words if w in q_lower]
    if found_mood:  result["mood_text"]  = " ".join(found_mood)
    if found_theme: result["theme_text"] = " ".join(found_theme)

    return result


def _fuzzy_match(name: str, movies_df: pd.DataFrame):
    name_lower = name.lower().strip()
    if not name_lower:
        return None
    try:
        titles_lower = movies_df["title"].fillna("").str.lower()
        exact = movies_df[titles_lower.str.startswith(name_lower, na=False)]
        if not exact.empty:
            return exact.iloc[0]["title"]
        partial = movies_df[titles_lower.str.contains(re.escape(name_lower), na=False)]
        if not partial.empty:
            return partial.iloc[0]["title"]
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════
#  VECTOR BUILDERS  (novelty techniques)
# ══════════════════════════════════════════════════════════════
def _negation_vector(free_text, negations, model, alpha=0.45):
    """N1: Subtract negated concepts from query vector."""
    base = model.encode(free_text, normalize_embeddings=True)
    if not negations:
        return base
    neg_vecs = np.array([model.encode(n, normalize_embeddings=True) for n in negations])
    result   = base - alpha * neg_vecs.mean(axis=0)
    norm     = np.linalg.norm(result)
    return result / norm if norm > 1e-8 else base

def _anchor_delta_vector(anchor_title, deltas, free_text, movies_df, text_emb, model):
    """N2: Shift from anchor film toward modifier adjectives."""
    idx_series = movies_df[movies_df["title"] == anchor_title].index
    if len(idx_series) and idx_series[0] < len(text_emb):
        anchor_vec = text_emb[idx_series[0]]
    else:
        anchor_vec = model.encode(anchor_title, normalize_embeddings=True)
    delta_vec = model.encode(" ".join(deltas), normalize_embeddings=True) if deltas else np.zeros_like(anchor_vec)
    free_vec  = model.encode(free_text or anchor_title, normalize_embeddings=True)
    result    = anchor_vec + 0.5 * delta_vec + 0.3 * free_vec
    norm      = np.linalg.norm(result)
    return result / norm if norm > 1e-8 else anchor_vec

def _mood_theme_vector(mood_text, theme_text, model, mood_w=0.35, theme_w=0.65):
    """N3: Separately encode mood tone and content theme."""
    mood_enc  = model.encode(f"emotional tone: {mood_text}",  normalize_embeddings=True)
    theme_enc = model.encode(f"story about: {theme_text}",    normalize_embeddings=True)
    result    = mood_w * mood_enc + theme_w * theme_enc
    norm      = np.linalg.norm(result)
    return result / norm if norm > 1e-8 else theme_enc

def _build_embeddings(candidates: pd.DataFrame, model) -> np.ndarray:
    """Build rich runtime embeddings: title×2 + genres + overview."""
    texts = []
    for _, row in candidates.iterrows():
        title    = str(row.get("title",    "") or "")
        genres   = str(row.get("genres",   "") or "").replace("|", " ")
        overview = str(row.get("overview", "") or "")[:350]
        tagline  = str(row.get("tagline",  "") or "")
        combined = f"{title} {title} {genres} {overview} {tagline}"
        texts.append(" ".join(combined.split()) or title)
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=64)

def _diversity_filter(df: pd.DataFrame, embeddings: np.ndarray,
                      k: int = 10, threshold: float = 0.88) -> pd.DataFrame:
    """N6: MMR-style diversity — skip movies too similar to already-selected."""
    selected, sel_embs = [], []
    for pos, (_, row) in enumerate(df.iterrows()):
        if len(selected) >= k or pos >= len(embeddings):
            break
        curr = embeddings[pos].reshape(1, -1)
        if not any(
            cosine_similarity(curr, e.reshape(1, -1))[0][0] > threshold
            for e in sel_embs
        ):
            selected.append(row)
            sel_embs.append(embeddings[pos])
    return pd.DataFrame(selected)


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def ultra_filter_recommend(query: str, context: dict,
                           exclude_genres_extra: list = None,
                           mood_theme_weights: tuple = (0.35, 0.65),
                           top_n: int = 10) -> tuple:
    model          = get_model()
    movies_df      = (context["movies"]
                      .drop_duplicates(subset="movieId")
                      .reset_index(drop=True))
    text_embeddings = context["text_embeddings"]
    query_lower    = query.lower()

    parsed      = parse_query(query, movies_df)
    all_exclude = list(set(parsed["exclude_genres"] + (exclude_genres_extra or [])))

    # ── STAGE 1: HARD FILTERS ────────────────────────────────
    candidates = movies_df.copy()

    # Explicit genre exclusions (from UI + parsed "no X")
    for g in all_exclude:
        pat        = GENRE_ALIASES.get(g.lower(), g)
        candidates = candidates[
            ~candidates["genres"].str.contains(pat, case=False, na=False)
        ]

    # Negative content auto-exclusion
    for neg_word, genres_drop in NEGATIVE_CONTENT.items():
        if neg_word in query_lower and genres_drop:
            if not any(pos in query_lower for pos in genres_drop):
                candidates = candidates[
                    ~candidates["genres"].str.contains(
                        "|".join(genres_drop), case=False, na=False)
                ]

    # Strong genre inclusion — but SKIP keywords that appear after a negation word
    negation_words_in_query = set(
        re.findall(r"(?:no|not|without|avoid|exclude)\s+(\w+)", query_lower)
    )
    detected_inc = [
        pat for kw, pat in INCLUSION_KEYWORDS.items()
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower)
        and kw not in negation_words_in_query
        and kw.replace("-","") not in negation_words_in_query
    ]
    if detected_inc:
        inc_pat  = "|".join(detected_inc)
        filtered = candidates[
            candidates["genres"].str.contains(inc_pat, case=False, na=False)
        ]
        if len(filtered) >= 20:
            candidates = filtered
        elif len(filtered) >= 5:
            extra      = candidates[~candidates.index.isin(filtered.index)].head(60)
            candidates = pd.concat([filtered, extra], ignore_index=True)

    # Era filter
    if parsed["era_range"]:
        yr_low, yr_high = parsed["era_range"]
        yr_series = None
        if "release_date" in candidates.columns:
            yr_series = pd.to_datetime(
                candidates["release_date"], errors="coerce"
            ).dt.year.values
        elif "release_year" in candidates.columns:
            yr_series = pd.to_numeric(
                candidates["release_year"], errors="coerce"
            ).values
        if yr_series is not None:
            mask = (yr_series >= yr_low) & (yr_series <= yr_high)
            era_filtered = candidates[mask]
            if len(era_filtered) >= 10:
                candidates = era_filtered

    # Safety floor
    if len(candidates) < 50:
        candidates = movies_df.copy()

    # ── STAGE 2: SEMANTIC SEARCH ─────────────────────────────
    primary_vec = model.encode(
        f"movie recommendation: {query}", normalize_embeddings=True
    )

    # Cap candidates for speed
    if len(candidates) > 3000:
        candidates = candidates.head(3000)
    candidates = candidates.reset_index(drop=True)

    # Prefer runtime embeddings (title+genres+overview) if overview available
    has_overview = (
        "overview" in candidates.columns
        and candidates["overview"].notna().sum() > 10
    )
    if has_overview:
        cand_emb = _build_embeddings(candidates, model)
    else:
        # Fall back to precomputed embeddings by position
        cand_indices = [
            movies_df.index[movies_df["movieId"] == mid].tolist()
            for mid in candidates["movieId"]
        ]
        valid_pos = [
            (i, idxs[0])
            for i, idxs in enumerate(cand_indices)
            if idxs and idxs[0] < len(text_embeddings)
        ]
        if not valid_pos:
            return pd.DataFrame(), parsed
        cand_rows    = [p[0] for p in valid_pos]
        emb_rows     = [p[1] for p in valid_pos]
        candidates   = candidates.iloc[cand_rows].reset_index(drop=True)
        cand_emb     = text_embeddings[emb_rows]

    scores = cosine_similarity(primary_vec.reshape(1, -1), cand_emb)[0]
    candidates = candidates.copy()
    candidates["_score"] = scores

    # ── STAGE 3: BOOSTING ────────────────────────────────────
    # 3a. Attribute keyword boosts
    for kw, (genres_list, boost_val) in ATTRIBUTE_BOOSTS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
            mask = candidates["genres"].str.contains(
                "|".join(genres_list), case=False, na=False
            )
            candidates.loc[mask, "_score"] += boost_val

    # 3b. N1 — Negation penalty
    if parsed["negations"]:
        neg_vec    = _negation_vector(query, parsed["negations"], model)
        neg_scores = cosine_similarity(neg_vec.reshape(1,-1), cand_emb)[0]
        candidates["_score"] -= 0.20 * np.clip(neg_scores, 0, 1)

    # 3c. N2 — Anchor + delta boost
    if parsed["anchor_movie"]:
        anc_vec    = _anchor_delta_vector(
            parsed["anchor_movie"], parsed["anchor_delta"], "",
            movies_df, text_embeddings, model,
        )
        anc_scores = cosine_similarity(anc_vec.reshape(1,-1), cand_emb)[0]
        candidates["_score"] += 0.30 * anc_scores

    # 3d. N3 — Mood-theme decoupled boost
    if parsed["mood_text"] and parsed["theme_text"]:
        mood_w, theme_w = mood_theme_weights
        mt_vec    = _mood_theme_vector(
            parsed["mood_text"], parsed["theme_text"], model, mood_w, theme_w
        )
        mt_scores = cosine_similarity(mt_vec.reshape(1,-1), cand_emb)[0]
        candidates["_score"] += 0.20 * mt_scores

    # 3e. Recency boost
    yr_map: dict = {}
    if "release_date" in movies_df.columns:
        yr_s   = pd.to_datetime(movies_df["release_date"], errors="coerce").dt.year
        yr_map = dict(zip(movies_df["movieId"], yr_s))
    elif "release_year" in movies_df.columns:
        yr_map = dict(zip(movies_df["movieId"],
                          pd.to_numeric(movies_df["release_year"], errors="coerce")))

    def _yr_boost(mid):
        y = yr_map.get(mid, np.nan)
        if pd.isna(y): return 0.0
        if y >= 2022:  return 0.10
        if y >= 2020:  return 0.07
        if y >= 2016:  return 0.04
        return 0.0
    candidates["_score"] += candidates["movieId"].apply(_yr_boost)

    # 3f. Popularity
    if "popularity" in candidates.columns:
        pop = pd.to_numeric(candidates["popularity"], errors="coerce").fillna(0)
        rng = pop.max() - pop.min()
        if rng > 0:
            candidates["_score"] += 0.04 * (pop - pop.min()) / (rng + 1e-8)

    # Clip and sort
    candidates["_score"] = candidates["_score"].clip(lower=0).fillna(0)
    
    # Remove the anchor movie itself from results (user already knows it)
    if parsed.get("anchor_movie"):
        candidates = candidates[
            candidates["title"].str.lower() != parsed["anchor_movie"].lower()
        ]
    
    candidates = candidates.sort_values("_score", ascending=False)

    # N6 — diversity filter
    results = _diversity_filter(candidates, cand_emb, k=top_n)
    results = results.copy().reset_index(drop=True)

    # Build reason tags
    def build_reason(row):
        parts = []
        if parsed.get("anchor_movie"):
            parts.append(f"Similar to {parsed['anchor_movie']}")
        if parsed.get("negations"):
            parts.append(f"Avoids: {', '.join(parsed['negations'][:2])}")
        if parsed.get("era_range"):
            parts.append(f"Era: {parsed['era_range'][0]}-{parsed['era_range'][1]}")
        genres_str = str(row.get("genres",""))
        if genres_str:
            g_list = [g.strip() for g in genres_str.split("|") if g.strip()][:3]
            parts.append("Genre: " + ", ".join(g_list).lower())
        if parsed.get("mood_text"):
            parts.append(f"Mood: {parsed['mood_text']}")
        return " · ".join(parts) if parts else "Strong semantic match"

    results["reason"] = results.apply(build_reason, axis=1)

    if "poster_path" not in results.columns:
        results["poster_path"] = None

    keep = ["movieId","title","genres","poster_path","_score","reason"]
    available = [c for c in keep if c in results.columns]
    return results[available].rename(columns={"_score": "score"}), parsed


# ══════════════════════════════════════════════════════════════
#  LLM EXPLANATION
# ══════════════════════════════════════════════════════════════
def generate_ultra_explanation(movie_title: str, genres: str, parsed: dict) -> str:
    """One-sentence explanation. Tries Groq, falls back to template."""
    try:
        import os
        from groq import Groq
        anchor    = parsed.get("anchor_movie","")
        negations = parsed.get("negations",[])
        mood      = parsed.get("mood_text","")
        theme     = parsed.get("theme_text","")
        era       = parsed.get("era_range")
        ctx_parts = []
        if anchor:    ctx_parts.append(f"similar to {anchor}")
        if negations: ctx_parts.append(f"avoiding {', '.join(negations[:2])}")
        if mood:      ctx_parts.append(f"with a {mood} tone")
        if theme:     ctx_parts.append(f"themed around {theme}")
        if era:       ctx_parts.append(f"from the {era[0]}-{era[1]} era")
        ctx_str = ", ".join(ctx_parts) if ctx_parts else "the user's query"
        prompt  = (
            f"In exactly one sentence, explain why '{movie_title}' ({genres}) "
            f"is a great recommendation for someone looking for a movie {ctx_str}. "
            f"Be specific. Do not start with I."
        )
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=100,
        )
        reply = response.choices[0].message.content.strip()
        if reply:
            return reply
    except Exception:
        pass

    # Template fallback
    genre_list = [g.strip() for g in str(genres).split("|") if g.strip()]
    genre_str  = ", ".join(genre_list[:3]).lower() if genre_list else "this genre"
    anchor     = parsed.get("anchor_movie","")
    mood       = parsed.get("mood_text","")
    if anchor:
        return (f"Recommended because it shares the spirit of {anchor} "
                f"while offering its own distinct {genre_str} experience.")
    if mood:
        return (f"A strong match for your {mood} mood — "
                f"this {genre_str} film closely fits your request.")
    return (f"This {genre_str} film closely matches your query "
            f"based on semantic similarity and genre alignment.")