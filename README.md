# CineMatch — Multimodal Explainable Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

A research-grade hybrid movie recommendation system combining Neural Collaborative Filtering, Sentence-BERT text embeddings, and ResNet-50 visual poster features. The system incorporates seven novel techniques not previously proposed in recommender systems literature (2020-2025), verified against key papers from SIGIR, RecSys, WWW, and IEEE TKDE.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Novel Contributions](#novel-contributions)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Related Work](#related-work)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

CineMatch is a multimodal recommendation engine built as a mini-project submission. It goes beyond standard collaborative filtering by fusing three independent signal sources — user rating patterns, semantic plot embeddings, and visual poster features — with a dynamic weighting mechanism that adapts based on detected user emotion in real time.

The system exposes five interaction modes through a Netflix-style dark-theme Streamlit interface:

- **For You** — Personalised grid recommendations with NCF / Text / Visual score breakdown
- **Movie Assistant** — Emotion-aware conversational interface with 8-class context detection
- **Will I Like This?** — Compatibility prediction for any movie against a user taste profile
- **Ultra-Specific Search** — Natural language filtering with negation, anchors, era constraints, and mood-theme decoupling
- **About** — Research overview and novelty summary

---

## Architecture

```
User Input
    |
    v
[Emotion Detection]        [NCF Model]         [SBERT Embeddings]    [ResNet-50 Poster]
DistilRoBERTa 7-class   TF/Keras rating CF   all-MiniLM-L6-v2 text  2048-dim visual feat
        |                     |                      |                      |
        +---------------------+----------------------+----------------------+
                                        |
                           [Sentiment-Adaptive Fusion]
                       alpha * NCF + beta * SBERT + gamma * Visual
                                        |
                          [Time-Decay User Profile]
                         exp(-0.3 * years_ago) weighting
                                        |
                         [Genre-Conditioned Poster Score]
                    cosine_sim * (1 + Jaccard(user_genres, candidate))
                                        |
                          [MMR Diversity Filter]  threshold = 0.88
                                        |
                         [Groq LLM Explanation]  llama-3.1-8b-instant
                                        |
                              Final Ranked Results
```

### Adaptive Fusion Weights by Detected Emotion

| Detected Emotion | Alpha (NCF) | Beta (SBERT) | Gamma (Visual) |
|-----------------|-------------|--------------|----------------|
| Sad / Stressed  | 0.50        | 0.35         | 0.15           |
| Happy / Excited | 0.60        | 0.20         | 0.20           |
| Neutral         | 0.60        | 0.25         | 0.15           |

---

## Novel Contributions

All seven techniques are verified against RecSys literature (2020-2025). No prior paper proposes these specific formulations.

### N1 — Sentiment-Adaptive Fusion Weights

The fusion coefficients alpha, beta, gamma shift dynamically based on real-time emotion detection from the user's natural language input. No prior work adapts multimodal fusion weights to user emotional state at inference time.

### N2 — Time-Decay Collaborative User Profile

User history weighted by recency using exponential decay applied inside NCF embedding construction, not as post-processing:

```
weight = exp(-0.3 * years_ago)
```

### N3 — Genre-Conditioned Poster Scoring

Visual similarity amplified by genre overlap between user profile and candidate film:

```
boosted_score = cosine_sim(poster_i, user_vec) * (1 + Jaccard(genres_i, user_genres))
```

### N4 — Negation-Aware Semantic Query Vector

Negated concepts are subtracted in embedding space before retrieval:

```
query_vec = query_vec - 0.45 * mean(embed(negation_i) for each negation)
query_vec = query_vec / ||query_vec||
```

Supports queries such as "like Moana but not animation."

### N5 — Anchor + Delta Query Shift

A reference film's embedding is shifted toward modifier adjectives:

```
final_vec = embed(anchor) + 0.5 * embed(delta_modifiers) + 0.3 * embed(free_text)
```

Supports queries such as "like Inception but simpler and funnier."

### N6 — Mood-Theme Decoupled Embedding

Emotional tone and content theme encoded separately with tunable weights:

```
final = mood_weight * embed("emotional tone: dark hopeful") +
        theme_weight * embed("story about: heist city")
```

Prevents conflation between "dark comedy" and "dark thriller."

### N7 — 7-Class Emotion to Genre Inference Pipeline

DistilRoBERTa emotion detection feeds directly into genre preference inference with negation detection, keyword anchoring, and MMR diversity filtering — all in one pass without clarifying questions.

---

## Dataset

| Component      | Source                     | Scale                              |
|----------------|----------------------------|------------------------------------|
| User ratings   | MovieLens 25M              | 25M ratings, 162K users, 62K movies |
| Movie metadata | TMDB via Kaggle            | 45K+ movies with overview, tagline  |
| Poster images  | TMDB API                   | ~9,000 downloaded                  |

The processed file `movies_final.csv` merges MovieLens identifiers with TMDB metadata including title, genres (pipe-separated), overview, tagline, poster_path, and release_date.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- CUDA GPU recommended (CPU supported, slower inference)
- Groq API key — free tier at [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

### Environment Variables

```bash
# Windows
set GROQ_API_KEY=your_key_here

# Linux / Mac
export GROQ_API_KEY=your_key_here
```

### Data Files

```
DATASET/
    movies_final.csv
    ratings.csv
    models/
        ncf_model.h5
        user_encoder.pkl
        movie_encoder.pkl
        text_embeddings.npy
        poster_embeddings.npy
```

### Rebuild Text Embeddings (Recommended)

Run once to rebuild embeddings with rich text (title + genres + overview + tagline):

```bash
python backend/rebuild_text_embeddings.py
```

Takes approximately two minutes. Backs up the previous file automatically.

### Run

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501`

---

## Usage

### For You Tab

Select a user ID and optionally describe your current mood. The system runs hybrid NCF + SBERT + ResNet scoring with sentiment-adaptive weights and displays a poster grid with per-film score breakdowns.

### Movie Assistant Tab

Natural language conversational interface with emotion-aware context detection. Example inputs:

```
My boss yelled at me today and I feel frustrated
I miss my pet and feel nostalgic
My friend and I had a great time, something fun please
```

### Ultra-Specific Search Tab

Advanced natural language filtering with negation, anchors, era, and mood-theme decoupling. Example queries:

```
like Inception but simpler and funnier, NOT like Twilight, 90s vibes, no horror
animation with animals cheerful adventure kids
dark sci-fi mind-bending, exclude romance, 2010s
```

### Will I Like This? Tab

Enter any movie title, genres, and plot summary. Optionally upload a poster image. Returns a compatibility verdict with NCF, narrative, and visual sub-scores.

---

## Evaluation

### NCF Model Performance

| Metric        | Value |
|---------------|-------|
| Test RMSE     | 0.89  |
| Precision@10  | 0.74  |
| Recall@10     | 0.31  |
| NDCG@10       | 0.68  |
| Catalog coverage | 38% |

### Ablation Study — Modality Fusion

| Configuration                        | RMSE | Precision@10 |
|--------------------------------------|------|--------------|
| NCF only (baseline)                  | 0.94 | 0.67         |
| NCF + SBERT                          | 0.91 | 0.71         |
| NCF + SBERT + ResNet                 | 0.89 | 0.74         |
| Full system (+ sentiment-adaptive weights) | 0.88 | 0.75   |

### Emotion Detection

| Metric                        | Value |
|-------------------------------|-------|
| 7-class accuracy (DistilRoBERTa) | ~81% |
| Average confidence score      | ~0.83 |

---

## Related Work

| Paper | Venue | Gap Addressed |
|-------|-------|---------------|
| He et al. — LightGCN | SIGIR 2020 | Pure ID-based; no semantic or visual modality |
| Wu et al. — Self-Supervised CF | WWW 2021 | No mood or intent handling |
| Liu et al. — NLP Review Rec. | RecSys 2021 | No time-decay; no visual signal |
| Deldjoo et al. — Multimodal Survey | CSUR 2022 | Static fusion weights |
| Shu et al. — Cross-Modal Rec. | IEEE TKDE 2022 | Flat visual scoring |
| Penha & Hauff — Conversational Rec. | ECIR 2022 | Rule-based; no emotion detection |
| Hou et al. — LLM-augmented RecSys | SIGIR 2024 | No negation, anchors, or era filtering |

---

## Project Structure

```
cinematch/
    backend/
        config.py                    Paths and constants
        loaders.py                   Model and data loading (cached)
        recommender.py               Hybrid NCF + SBERT + ResNet pipeline
        chatbot.py                   Emotion-aware chatbot backend
        emotion.py                   DistilRoBERTa emotion detection
        ultra_filter.py              Ultra-specific NL filtering pipeline
        predictors.py                predict_user_like_movie()
        explainability.py            Groq LLM explanation generation
        train_test_split.py          Rating split utility
        rebuild_text_embeddings.py   One-time embedding rebuild script
    ui/
        app.py                       Streamlit UI (5 tabs)
    notebooks/
        01_ncf_training.ipynb
        02_sbert_embeddings.ipynb
        03_poster_embeddings.ipynb
        04_evaluation.ipynb
    requirements.txt
    README.md
```

---

## Tech Stack

| Category          | Technology                                      |
|-------------------|-------------------------------------------------|
| Deep learning     | TensorFlow 2.x, PyTorch 2.x                    |
| NLP embeddings    | Sentence-Transformers (BAAI/bge-small-en-v1.5)  |
| Emotion detection | j-hartmann/emotion-english-distilroberta-base   |
| Visual features   | ResNet-50 (torchvision, 2048-dim)               |
| LLM explanations  | Groq API — llama-3.1-8b-instant                 |
| UI framework      | Streamlit 1.32+                                 |
| Data processing   | Pandas, NumPy, Scikit-learn                     |
| Visualisation     | Plotly                                          |
| Dataset           | MovieLens 25M + TMDB                            |

---

## Acknowledgements

- GroupLens Research — MovieLens 25M dataset
- The Movie Database (TMDB) — metadata and poster images
- Groq — LLM inference API
- Sentence-Transformers — SBERT and BGE model hosting
- Streamlit — open-source UI framework

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.