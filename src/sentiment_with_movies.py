import pandas as pd

MOVIES_PATH = "./data/movie.csv"
SCORES_PATH = "./data/movie_sentiment_scores.csv"
OUT = "./data/movies_with_sentiment.csv"

movies = pd.read_csv(MOVIES_PATH)
scores = pd.read_csv(SCORES_PATH)

merged = movies.merge(scores, on="movieId", how="left")

print("Total movies:", len(movies))
print("Movies with sentiment:", merged['sentiment_score'].notna().sum())

merged.to_csv(OUT, index=False)
print("Saved:", OUT)
