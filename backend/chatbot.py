from backend.emotion import detect_emotion, emotion_to_genres, detect_intent
from backend.recommender import recommend_movies_multimodal
from groq import Groq
import os
def format_mood(emotion):
    """
    Convert internal emotion labels to friendly UI messages
    """

    mood_map = {
        "joy": "Feeling good 😄",
        "sadness": "Tough day 😔",
        "anger": "Feeling frustrated 😤",
        "fear": "Feeling stressed 😟",
        "surprise": "Something unexpected 😮",
        "neutral": "Just browsing 🙂",
        "disgust": "Rough moment 🤨"
    }

    return mood_map.get(emotion, "Movie mood detected 🎬")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_dynamic_reply(user_message, emotion, genres):

    prompt = f"""
User message: {user_message}

Detected emotion: {emotion}

Recommended genres: {genres}

You are a friendly movie recommendation assistant.

Write a short empathetic response (2–3 sentences).
Do NOT list or suggest specific movie titles.
Just acknowledge the user's feelings and say that you found some movies that might help.

Keep the tone warm and supportive.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=100
    )

    reply = completion.choices[0].message.content
    reply = completion.choices[0].message.content.strip()

# remove extra line breaks
    reply = reply.replace("\n\n", "\n")

    # ensure reply ends properly
    if not reply.endswith((".", "!", "?")):
        reply += "."


    return reply


def chatbot_recommendation(user_id, message, context):

    # Detect emotion
    emotion, confidence = detect_emotion(message)

    # Detect intent
    intent, intent_genres = detect_intent(message)

    # Genres from emotion
    emotion_genres = emotion_to_genres(emotion)

    # Combine both
    genres = list(set(intent_genres + emotion_genres))

    # Get recommendations
    recommendations = recommend_movies_multimodal(
        user_id=user_id,
        context=context,
        top_n=20
    )

    # Genre filtering
    if genres:
        filtered = recommendations[
            recommendations["genres"].str.contains("|".join(genres), case=False)
        ]
    else:
        filtered = recommendations

    # Fallback if nothing found
    if filtered.empty:
        filtered = recommendations

    filtered = filtered.head(10)

    # Generate dynamic chatbot reply
    reply = generate_dynamic_reply(message, emotion, genres)

    return {
    "emotion": emotion,
    "mood_display": format_mood(emotion),
    "confidence": confidence,
    "genres": genres,
    "reply": reply,
    "recommendations": filtered
}