# ui/app.py
import sys
import os
import pandas as pd
import streamlit as st

# -------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.loaders import load_all
from backend.recommender import recommend_movies_multimodal
from backend.predictors import predict_user_like_movie
from backend.explainability import generate_llm_explanation

# -------------------------------------------------
# GLOBAL DARK THEME CSS (FULL APP)
# -------------------------------------------------
st.markdown("""
<style>

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Labels */
label {
    color: #e5e7eb !important;
}

/* Normal text */
p, span, div {
    color: #f1f5f9;
}

/* Inputs */
input, textarea {
    color: #ffffff !important;
}

/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 600;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #cbd5f5 !important;
}

</style>
""", unsafe_allow_html=True)




# -------------------------------------------------
# LOAD RESOURCES
# -------------------------------------------------
@st.cache_resource
def load_resources():
    return load_all()

context = load_resources()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable Multimodal Movie Recommendation",
    layout="wide"
)

st.title(" Explainable Multimodal Movie Recommendation System")

tabs = st.tabs([
    " Existing User",
    " Will I Like This Movie?",
    "ℹ About"
])

# =================================================
# TAB 1 — EXISTING USER
# =================================================
with tabs[0]:
    st.subheader("Recommendations for Existing Users")

    user_id = st.selectbox(
        "Select User ID",
        sorted(context["ratings"].userId.unique())
    )

    if st.button(" Recommend Movies"):
        recs = recommend_movies_multimodal(
            user_id=user_id,
            context=context,
            top_n=10
        )

        recent_titles = (
            context["movies"][
                context["movies"].movieId.isin(
                    context["ratings"][context["ratings"].userId == user_id].movieId
                )
            ]["title"].tolist()[:3]
        )

        for _, row in recs.iterrows():
            explanation = generate_llm_explanation({
                "recommended_movie": row["title"],
                "genres": row["genres"],
                "recent_movies": recent_titles
            })

            col_img, col_txt = st.columns([1, 3])

            with col_img:
                if "poster_path" in row and pd.notna(row["poster_path"]):
                    st.image(
                        f"https://image.tmdb.org/t/p/w500{row['poster_path']}",
                        use_container_width=True
                    )
                else:
                    st.caption("Poster not available")

            with col_txt:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genres">{row['genres']}</div>
                    <div class="explanation">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)

# =================================================
# TAB 2 — WILL I LIKE THIS MOVIE?
# =================================================
with tabs[1]:
    st.subheader("Will You Like This Movie?")

    user_id = st.selectbox(
        "Select User ID",
        sorted(context["ratings"].userId.unique()),
        key="predict_user"
    )

    title = st.text_input("Movie Title")
    genres = st.text_input("Genres (comma-separated)")
    plot = st.text_area("Plot / Description")

    st.markdown("###  Optional: Upload Movie Poster")
    uploaded_poster = st.file_uploader(
        "Upload poster",
        type=["jpg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_poster:
        st.image(uploaded_poster, width=240)

    if st.button(" Predict Preference"):
        movie_text = f"{title} {genres} {plot}"

        result = predict_user_like_movie(
            user_id=user_id,
            movie_text=movie_text,
            context=context,
            uploaded_poster=uploaded_poster
        )

        verdict = result["verdict"]
        confidence = result["confidence"]

        if verdict == "like":
            st.markdown(f"<div class='badge-like'> Likely to like ({confidence})</div>", unsafe_allow_html=True)
        elif verdict == "maybe":
            st.markdown(f"<div class='badge-maybe'> Might like ({confidence})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge-unlikely'> Unlikely ({confidence})</div>", unsafe_allow_html=True)

        st.metric("Final Similarity Score", f"{result['final_score']:.2f}")

        explanation = generate_llm_explanation({
            "recommended_movie": title,
            "genres": genres,
            "recent_movies": [],
            "verdict": verdict,
            "used_poster": result["used_poster"]
        })

        st.markdown(f"""
        <div class="movie-card">
            <div class="explanation">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)

# =================================================
# TAB 3 — ABOUT
# =================================================
with tabs[2]:
    st.markdown("""
    ## System Overview

    - Neural Collaborative Filtering  
    - Text Embeddings (SBERT)  
    - Poster Embeddings (ResNet-50)  
    - Multimodal Fusion  
    - LLM-based Explainability  

    **A production-grade, explainable recommender system.**
    """)
