from backend.loaders import load_all
from backend.evaluation import evaluate_recommender

context = load_all()

evaluate_recommender(context, k=10)