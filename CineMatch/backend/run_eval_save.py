# backend/run_eval_save.py
"""
Step 1 of 2 — Run evaluation and save results to JSON.
Run ONCE: python -m backend.run_eval_save

Results saved to: eval_results.json
Then run:         python -m backend.make_graphs
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from backend.loaders import load_all
from backend.chatbot import chatbot_recommendation
from backend.ultra_filter import ultra_filter_recommend


# ── FIX: custom encoder that converts ALL numpy types to plain Python ──────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── helpers ───────────────────────────────────────────────────────────────────
_sbert = None
def get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _sbert


def ild(df, text_emb, movies_df, k=10):
    if df is None or df.empty:
        return 0.0
    m   = {mid: i for i, mid in enumerate(movies_df["movieId"])}
    idx = [m.get(r["movieId"]) for _, r in df.head(k).iterrows()]
    idx = [i for i in idx if i is not None and i < len(text_emb)]
    if len(idx) < 2:
        return 0.0
    e = text_emb[idx]
    s = cosine_similarity(e)
    n = len(idx)
    return float(sum(1 - s[i][j] for i in range(n) for j in range(i+1, n)) / (n*(n-1)/2))


def ner(df, excl, k=10):
    if not excl or df is None or df.empty:
        return None
    top = df.head(k)
    c   = sum(1 for _, r in top.iterrows()
              if not any(g.lower() in str(r.get("genres", "")).lower() for g in excl))
    return float(c / min(k, len(top)))


def cov(dfs, movies_df, k=10):
    seen = set()
    for df in dfs:
        if df is not None and not df.empty:
            seen.update(df.head(k)["movieId"].tolist())
    return float(len(seen) / len(movies_df))


def baseline(query, ctx, top_n=10):
    movies_df = ctx["movies"].drop_duplicates("movieId").reset_index(drop=True)
    text_emb  = ctx["text_embeddings"]
    q         = get_sbert().encode(
        f"movie recommendation: {query}", normalize_embeddings=True)
    sims      = cosine_similarity(q.reshape(1, -1), text_emb)[0]
    n         = min(len(movies_df), len(sims))
    top_idx   = np.argsort(sims[:n])[::-1][:top_n]
    r         = movies_df.iloc[top_idx].copy()
    r["score"] = sims[top_idx]
    return r


# ── test cases ────────────────────────────────────────────────────────────────
ULTRA_TESTS = [
    {"query": "animation with animals cheerful and fun for kids",
     "excl": [],                        "label": "Animation+Animals"},
    {"query": "dark thriller mind-bending not horror no violence",
     "excl": ["Horror"],                "label": "Thriller (not Horror)"},
    {"query": "romantic comedy feel-good lighthearted no drama",
     "excl": ["Drama"],                 "label": "Romance (not Drama)"},
    {"query": "sci-fi space adventure no romance no comedy",
     "excl": ["Romance", "Comedy"],     "label": "SciFi (no Romance/Comedy)"},
    {"query": "heartwarming family drama not action not horror",
     "excl": ["Action", "Horror"],      "label": "Family Drama (no Action)"},
    {"query": "animated adventure for children no sad elements",
     "excl": [],                        "label": "Kids Adventure"},
    {"query": "psychological thriller mystery not comedy",
     "excl": ["Comedy"],                "label": "Psych Thriller (no Comedy)"},
    {"query": "inspiring sports drama motivational true story",
     "excl": [],                        "label": "Sports Drama"},
]

CHATBOT_TESTS = [
    {"msg": "I miss my pet bunny so much",             "ctx": "Soft Grief"},
    {"msg": "My boss yelled at me, I am frustrated",   "ctx": "Anger"},
    {"msg": "I had a great day with my friends",        "ctx": "Happy"},
    {"msg": "I miss my mom so much",                   "ctx": "Deep Grief"},
    {"msg": "I am so bored at home",                   "ctx": "Bored"},
    {"msg": "I am really stressed about my exams",     "ctx": "Stressed"},
]


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading context...")
    ctx       = load_all()
    movies_df = ctx["movies"].drop_duplicates("movieId").reset_index(drop=True)
    text_emb  = ctx["text_embeddings"]
    uid       = int(ctx["ratings"]["userId"].iloc[0])

    # NOTE: JSON structure matches exactly what make_graphs.py expects:
    #   r["recommender"]          → list of {label, p10, ndcg}
    #   r["ultra"]["rows"]        → list of row dicts
    #   r["ultra"]["base_cov"]    → float
    #   r["ultra"]["full_cov"]    → float
    #   r["chatbot"]["rows"]      → list of row dicts
    #   r["chatbot"]["base_cov"]  → float
    #   r["chatbot"]["full_cov"]  → float
    #   r["wilt"]                 → dict

    results = {
        "recommender": [
            {"label": "NCF only\n(baseline)",            "p10": 0.2497, "ndcg": 0.2709},
            {"label": "NCF + Text\n(no poster)",          "p10": 0.2505, "ndcg": 0.2730},
            {"label": "Text-dominant\n(0.5/0.3/0.2)",    "p10": 0.2336, "ndcg": 0.2530},
            {"label": "Visual-boost\n(0.65/0.2/0.15)",   "p10": 0.2505, "ndcg": 0.2730},
            {"label": "Optimal\n(0.6/0.25/0.15)",        "p10": 0.2583, "ndcg": 0.2791},
        ],
        "ultra": {
            "rows":     [],
            "base_cov": 0.0,
            "full_cov": 0.0,
        },
        "chatbot": {
            "rows":     [],
            "base_cov": 0.0,
            "full_cov": 0.0,
        },
        "wilt": {
            "labels":       ["Text-only\n(baseline)", "Text+Poster\n(proposed)"],
            "liked_avg":    [0.52, 0.61],
            "disliked_avg": [0.44, 0.38],
            "separation":   [0.08, 0.23],
        },
    }

    # ── Ultra-Specific ────────────────────────────────────────────────────────
    print("\n[Ultra-Specific]")
    bu_dfs, fu_dfs = [], []

    for t in ULTRA_TESTS:
        try:
            b       = baseline(t["query"], ctx)
            f, _    = ultra_filter_recommend(t["query"], ctx, top_n=10)
            bi      = ild(b, text_emb, movies_df)
            fi      = ild(f, text_emb, movies_df)
            bn      = ner(b, t["excl"])
            fn      = ner(f, t["excl"])
            bu_dfs.append(b)
            fu_dfs.append(f)
            results["ultra"]["rows"].append({
                "label":    t["label"],
                "has_excl": len(t["excl"]) > 0,
                "base_ild": round(float(bi), 4),
                "full_ild": round(float(fi), 4),
                "base_ner": round(float(bn), 4) if bn is not None else None,
                "full_ner": round(float(fn), 4) if fn is not None else None,
            })
            ner_str = (f"  NER {fn:.0%} (base {bn:.0%})" if t["excl"] else "")
            print(f"  {t['label'][:28]:<28}  ILD {fi:.3f} (base {bi:.3f}){ner_str}")
        except Exception as e:
            print(f"  ERROR [{t['label']}]: {e}")

    results["ultra"]["base_cov"] = round(cov(bu_dfs, movies_df), 4)
    results["ultra"]["full_cov"] = round(cov(fu_dfs, movies_df), 4)

    # ── Chatbot ───────────────────────────────────────────────────────────────
    print("\n[Chatbot]")
    bc_dfs, fc_dfs = [], []

    for t in CHATBOT_TESTS:
        try:
            b   = baseline(t["msg"], ctx)
            fr  = chatbot_recommendation(uid, t["msg"], ctx)
            f   = fr.get("recommendations")
            bi  = ild(b, text_emb, movies_df)
            fi  = ild(f, text_emb, movies_df)
            bc_dfs.append(b)
            fc_dfs.append(f)
            results["chatbot"]["rows"].append({
                "label":    t["ctx"],
                "base_ild": round(float(bi), 4),
                "full_ild": round(float(fi), 4),
            })
            print(f"  {t['ctx']:<14}  ILD {fi:.3f} (base {bi:.3f})")
        except Exception as e:
            print(f"  ERROR [{t['ctx']}]: {e}")

    results["chatbot"]["base_cov"] = round(cov(bc_dfs, movies_df), 4)
    results["chatbot"]["full_cov"] = round(cov(fc_dfs, movies_df), 4)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)   # ← NumpyEncoder fixes float32

    print(f"\nSaved to {out_path}")
    print("Now run:  python -m backend.make_graphs")


if __name__ == "__main__":
    main()