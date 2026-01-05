import pandas as pd

INPUT = "./data/reviews_with_lstm_sentiment.csv"
OUTPUT = "./data/movie_lstm_sentiment_scores.csv"

df = pd.read_csv(INPUT)

movie_scores = (
    df.groupby("movieId")["lstm_prob_positive"]
    .mean()
    .reset_index()
)

movie_scores.columns = ["movieId", "lstm_sentiment_score"]

movie_scores.to_csv(OUTPUT, index=False)
