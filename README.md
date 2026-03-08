# Multimodal Explainable Movie Recommendation System

A hybrid AI-based movie recommendation system that combines collaborative filtering, semantic text embeddings, visual poster embeddings, and conversational interfaces to deliver personalized movie suggestions.

The system integrates deep learning models, multimodal feature fusion, and a sentiment-aware chatbot to generate context-aware recommendations.

---

# Project Overview

Traditional recommender systems rely solely on user interaction data, which leads to challenges such as cold-start problems and lack of explainability.

This project addresses these issues by combining multiple recommendation strategies:

- Collaborative filtering using **Neural Collaborative Filtering (NCF)**
- Semantic similarity using **Sentence-BERT text embeddings**
- Visual similarity using **ResNet-based poster embeddings**
- **Weighted multimodal fusion** for final ranking
- **Emotion-aware conversational recommendations** using a chatbot
- Interactive **Streamlit user interface**

The system recommends movies based on:

- user rating history
- semantic similarity of plots
- visual similarity of posters
- conversational mood-based queries

---

# System Architecture

User Interaction
↓
Emotion Detection / Chatbot
↓
Multimodal Recommendation Engine

├── Neural Collaborative Filtering
├── Text Embeddings (Sentence-BERT)
├── Poster Embeddings (ResNet)

↓
Weighted Late Fusion
↓
Ranked Movie Recommendations

---

# Dataset

The system integrates multiple data sources.

## MovieLens 20M Dataset
Used for collaborative filtering.

Features:
- userId
- movieId
- rating
- timestamp

## TMDB Movie Dataset
Used for metadata and posters.

Features:
- title
- genres
- overview
- poster images
- release year

Recent movies were added using the **TMDB API** to extend the dataset with modern releases.

---

# Core Components

## 1. Neural Collaborative Filtering (NCF)

Learns nonlinear user–movie interactions.

Architecture:

User Embedding
Movie Embedding
↓
Concatenation
↓
MLP Layers
↓
Predicted Rating


---

## 2. Text Embeddings

Movie plots and metadata are encoded using:

SentenceTransformer("all-MiniLM-L6-v2")


User preferences are represented by averaging embeddings of previously liked movies.

Similarity is computed using **cosine similarity**.

---

## 3. Poster Embeddings

Movie posters are processed using a pretrained **ResNet50 CNN**.

Pipeline:


User preferences are represented by averaging embeddings of previously liked movies.

Similarity is computed using **cosine similarity**.

---

## 3. Poster Embeddings

Movie posters are processed using a pretrained **ResNet50 CNN**.

Pipeline:

Poster Image
↓
Image Preprocessing
↓
ResNet Feature Extractor
↓
2048-Dimensional Embedding


Poster similarity is computed using cosine similarity.

---

## 4. Multimodal Fusion

Final recommendation score:

Score =
α × NCF Score

β × Text Similarity

γ × Poster Similarity


Best performing weights:

α = 0.6
β = 0.25
γ = 0.15


---

## 5. Cold Start Handling

New movies without rating data are handled using content-based similarity.

Score =
0.7 × Text Similarity

0.3 × Poster Similarity


This allows the system to recommend newly released movies.

---

## 6. Emotion-Aware Chatbot

Users can interact with the system conversationally.

Example queries:

"I had a stressful day"
"I want something funny"
"I feel sad today"


Pipeline:

User Message
↓
Emotion Detection
↓
Genre Mapping
↓
Movie Recommendation
↓
LLM Generated Response


LLM responses are generated using **Groq API (Llama 3.1)**.

---

# Evaluation

The model was evaluated using a time-based train-test split.

Metrics used:

- Precision@10
- Recall@10
- NDCG@10

Example results:

| Configuration | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| α=0.6 β=0.25 γ=0.15 | 0.258 | 0.033 | 0.279 |

---

# User Interface

The system includes a Streamlit-based web interface featuring:

- Personalized recommendations
- Mood-based chatbot interaction
- Movie similarity exploration
- Poster visualization

---

# Project Structure
project/

backend/
loaders.py
recommender.py
predictors.py
chatbot.py
emotion.py

ui/
app.py

data/
movies.csv
ratings.csv

models/
ncfmodel.keras


---

# Technologies Used

- Python
- TensorFlow / Keras
- SentenceTransformers
- ResNet50
- Streamlit
- Groq API
- Scikit-learn
- Pandas / NumPy

---

# Future Improvements

- Ultra-specific semantic recommendations
- Improved UI with movie card layouts
- Real-time movie updates via TMDB
- Reinforcement learning-based ranking

---

# Author

Your Name  
GitHub: (https://github.com/madhurabashinims/Movie-recommendation-system)
