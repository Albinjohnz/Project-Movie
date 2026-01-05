import pandas as pd

MOVIES_BASE = "./data/movies_with_sentiment.csv"
BERT_SCORE_PATH = "./data/movie_bert_sentiment_scores.csv"
OUT = "./data/movies_with_bert_only.csv"

movies = pd.read_csv(MOVIES_BASE)

if "sentiment_score" in movies.columns:
    movies = movies.drop(columns=["sentiment_score"])

bert = pd.read_csv(BERT_SCORE_PATH)

merged = movies.merge(bert, on="movieId", how="left")
merged["bert_sentiment_score"] = merged["bert_sentiment_score"].fillna(0.5)

merged.to_csv(OUT, index=False)
