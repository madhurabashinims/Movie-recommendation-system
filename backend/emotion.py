# backend/emotion.py

from transformers import pipeline

# ── Emotion model ─────────────────────────────────────────────────────────
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
)

# ── Confidence threshold ──────────────────────────────────────────────────
# If the transformer is less than 45% confident, default to neutral
CONFIDENCE_THRESHOLD = 0.45

# ── Negation words ────────────────────────────────────────────────────────
NEGATION_WORDS = {"not", "no", "never", "don't", "doesn't", "nothing", "hardly", "barely"}

# ── Keyword → emotion map (expanded) ─────────────────────────────────────
keyword_map = {
    # sadness / low energy
    "stress":       "sadness",
    "stressed":     "sadness",
    "tired":        "sadness",
    "exhausted":    "sadness",
    "lonely":       "sadness",
    "alone":        "sadness",
    "depressed":    "sadness",
    "heartbreak":   "sadness",
    "heartbroken":  "sadness",
    "grief":        "sadness",
    "sad":          "sadness",
    "miserable":    "sadness",
    "exam":         "sadness",
    "exam stress":  "sadness",
    "breakup":      "sadness",
    "crying":       "sadness",

    # joy / positive
    "exciting":     "joy",
    "excited":      "joy",
    "funny":        "joy",
    "fun":          "joy",
    "happy":        "joy",
    "great":        "joy",
    "lighten":      "joy",
    "celebrate":    "joy",
    "celebration":  "joy",
    "promotion":    "joy",
    "party":        "joy",
    "motivated":    "joy",
    "pumped":       "joy",
    "energetic":    "joy",
    "upbeat":       "joy",

    # neutral / calm
    "relax":        "neutral",
    "relaxing":     "neutral",
    "chill":        "neutral",
    "bored":        "neutral",
    "boring":       "neutral",
    "lazy":         "neutral",
    "nostalgic":    "neutral",
    "casual":       "neutral",
    "weekend":      "neutral",

    # fear / anxiety
    "dark":         "fear",
    "anxious":      "fear",
    "anxiety":      "fear",
    "nervous":      "fear",
    "scared":       "fear",
    "worried":      "fear",
    "panic":        "fear",
    "creepy":       "fear",

    # anger
    "angry":        "anger",
    "frustrated":   "anger",
    "annoyed":      "anger",
    "rage":         "anger",
    "furious":      "anger",

    # surprise
    "surprised":    "surprise",
    "unexpected":   "surprise",
    "shocked":      "surprise",
    "mind-blown":   "surprise",
    "twist":        "surprise",
}

# ── Intent → genre map (expanded) ─────────────────────────────────────────
intent_keywords = {
    "relaxing":     ["Family", "Comedy", "Romance"],
    "dark":         ["Thriller", "Mystery", "Crime"],
    "funny":        ["Comedy"],
    "exciting":     ["Action", "Adventure"],
    "uplifting":    ["Comedy", "Family", "Animation"],
    "nostalgic":    ["Drama", "Romance", "Family"],
    "romantic":     ["Romance", "Drama"],
    "scary":        ["Horror", "Thriller"],
    "mind-bending": ["Sci-Fi", "Mystery", "Thriller"],
    "inspirational":["Drama", "Biography"],
    "adventure":    ["Adventure", "Action", "Fantasy"],
    "animated":     ["Animation", "Family"],
    "classic":      ["Drama", "Romance"],
}

# ── Emotion → genre map (expanded) ───────────────────────────────────────
emotion_genre_map = {
    "joy":      ["Comedy", "Adventure", "Animation", "Action"],
    "sadness":  ["Comedy", "Family", "Animation", "Romance"],
    "anger":    ["Comedy", "Action", "Thriller"],
    "fear":     ["Thriller", "Mystery", "Horror"],
    "surprise": ["Sci-Fi", "Mystery", "Thriller"],
    "neutral":  ["Drama", "Romance", "Documentary"],
    "disgust":  ["Comedy", "Drama"],
}


def _has_negation(text_lower: str, keyword: str) -> bool:
    """
    Return True if any negation word appears within 3 tokens before the keyword.
    e.g. "not funny" → True, "not in a funny mood" → True
    """
    tokens = text_lower.split()
    for i, token in enumerate(tokens):
        if keyword in token:
            window = tokens[max(0, i - 3): i]
            if any(neg in window for neg in NEGATION_WORDS):
                return True
    return False


def detect_emotion(text: str):
    """
    Detect emotion from text.

    Priority:
      1. Keyword override (with negation check)
      2. Transformer model (if confidence >= CONFIDENCE_THRESHOLD)
      3. Fallback to 'neutral'

    Returns (emotion_label: str, confidence: float)
    """
    text_lower = text.lower()

    # 1. Keyword scan — skip if negated
    for word, emotion in keyword_map.items():
        if word in text_lower and not _has_negation(text_lower, word):
            return emotion, 0.90

    # 2. Transformer
    result = emotion_model(text[:512])[0]   # clip to avoid token overflow
    label      = result["label"].lower()
    confidence = result["score"]

    if confidence >= CONFIDENCE_THRESHOLD:
        return label, confidence

    # 3. Low-confidence fallback
    return "neutral", confidence


def detect_intent(text: str):
    """
    Detect viewing intent from keyword match.

    Returns (intent: str | None, genres: list[str])
    """
    text_lower = text.lower()

    for intent, genres in intent_keywords.items():
        if intent in text_lower and not _has_negation(text_lower, intent):
            return intent, genres

    return None, []


def emotion_to_genres(emotion: str) -> list:
    """Map emotion label to a list of preferred genres."""
    return emotion_genre_map.get(emotion.lower(), ["Drama"])