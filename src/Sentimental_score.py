import pandas as pd

INPUT = "./data/reviews_with_sentiment.csv"
OUTPUT = "./data/movie_sentiment_scores.csv"

print("Loading labeled reviews...")
df = pd.read_csv(INPUT)

print("Total labeled reviews:", len(df))

# Convert neutral (2) to 0.5 to keep numeric meaning
df["sentiment_numeric"] = df["sentiment"].replace({
    1: 1.0,
    0: 0.0,
    2: 0.5
})

print("Computing movie sentiment scores...")
movie_scores = (
    df.groupby("movieId")["sentiment_numeric"]
    .mean()
    .reset_index()
)

movie_scores.columns = ["movieId", "sentiment_score"]

print("Movies with sentiment:", len(movie_scores))

movie_scores.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(movie_scores.head())
