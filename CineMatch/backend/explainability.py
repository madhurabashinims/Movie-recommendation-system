# backend/explainability.py

from groq import Groq

client = Groq()


def generate_llm_explanation(expl_data: dict) -> str:
    """
    Generates a human-friendly explanation for recommendations or predictions.

    expl_data may contain:
    - recommended_movie (str)
    - genres (str)
    - recent_movies (list[str])
    - verdict (optional: "like" | "maybe" | "dislike")
    """

    movie = expl_data.get("recommended_movie", "this movie")
    genres = expl_data.get("genres", "")
    recent = expl_data.get("recent_movies", [])
    verdict = expl_data.get("verdict")  # OPTIONAL

    # ---------- Build explanation context ----------
    recent_text = ""
    if recent:
        recent_text = (
            "You recently watched movies like "
            + ", ".join(recent[:3])
            + "."
        )

    verdict_text = ""
    if verdict is not None:
        verdict_text = (
            f"The system predicts that you are **{verdict.upper()}** to enjoy this movie."
        )

    # ---------- Prompt ----------
    prompt = f"""
You are an intelligent movie recommender assistant.

Write a natural, friendly explanation for a movie recommendation.

Movie: {movie}
Genres: {genres}

{recent_text}
{verdict_text}

Rules:
- Sound conversational and human
- Do NOT mention AI, models, embeddings, or scores
- Explain *why* the movie fits the user's taste
- 2–4 sentences only
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )

    return response.choices[0].message.content.strip()
