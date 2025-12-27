import pandas as pd

MOVIES_SENT_PATH = "./data/movies_with_sentiment.csv"
RATING_SCORE_PATH = "./data/movie_rating_scores.csv"
OUT = "./data/movies_with_sentiment_and_rating.csv"

movies = pd.read_csv(MOVIES_SENT_PATH)
ratings = pd.read_csv(RATING_SCORE_PATH)

merged = movies.merge(ratings[["movieId", "rating_score"]], on="movieId", how="left")
merged["rating_score"] = merged["rating_score"].fillna(0.5)  # neutral if no ratings

print("Total movies:", len(merged))
print("Movies with rating_score:", merged['rating_score'].notna().sum())

merged.to_csv(OUT, index=False)
print("Saved:", OUT)
