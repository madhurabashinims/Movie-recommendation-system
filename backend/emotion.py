from transformers import pipeline

# Emotion detection model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# keyword based mood detection
keyword_map = {
    "stress": "sadness",
    "tired": "sadness",
    "lonely": "sadness",
    "relax": "neutral",
    "relaxing": "neutral",
    "dark": "fear",
    "exciting": "joy",
    "funny": "joy",
    "lighten": "joy",
    "celebrate": "joy",
    "promotion": "joy",
    "exam": "sadness",
}

# intent keywords
intent_keywords = {
    "relaxing": ["Family", "Comedy", "Romance"],
    "dark": ["Thriller", "Mystery", "Crime"],
    "funny": ["Comedy"],
    "exciting": ["Action", "Adventure"],
    "uplifting": ["Comedy", "Family", "Animation"]
}


def detect_emotion(text):

    text_lower = text.lower()

    # check keyword overrides first
    for word, emotion in keyword_map.items():
        if word in text_lower:
            return emotion, 0.9

    # fallback to transformer
    result = emotion_model(text)[0]

    return result["label"], result["score"]


def detect_intent(text):

    text_lower = text.lower()

    for intent, genres in intent_keywords.items():
        if intent in text_lower:
            return intent, genres

    return None, []


def emotion_to_genres(emotion):

    mapping = {

        "joy": ["Comedy", "Adventure", "Animation"],

        "sadness": ["Comedy", "Family", "Animation"],

        "anger": ["Comedy"],

        "fear": ["Thriller", "Mystery"],

        "surprise": ["Sci-Fi", "Mystery"],

        "neutral": ["Drama", "Romance"]
    }

    return mapping.get(emotion, ["Drama"])