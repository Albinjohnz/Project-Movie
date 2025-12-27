import pandas as pd

RATING_PATH = "./data/rating.csv"   # change to ratings.csv if needed
OUT_PATH    = "./data/movie_rating_scores.csv"

print("Loading ratings...")
ratings = pd.read_csv(RATING_PATH, usecols=["movieId", "rating"])

print("Total ratings:", len(ratings))

movie_mean = ratings.groupby("movieId")["rating"].mean().reset_index()
movie_mean.columns = ["movieId", "avg_rating"]

# Normalize rating to 0–1
min_r = movie_mean["avg_rating"].min()
max_r = movie_mean["avg_rating"].max()
movie_mean["rating_score"] = (movie_mean["avg_rating"] - min_r) / (max_r - min_r)

print("Movies with ratings:", len(movie_mean))

movie_mean.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
