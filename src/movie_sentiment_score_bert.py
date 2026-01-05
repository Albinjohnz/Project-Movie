import pandas as pd

INPUT = "./data/reviews_with_bert_sentiment.csv"
OUTPUT = "./data/movie_bert_sentiment_scores.csv"

df = pd.read_csv(INPUT)

df["bert_sentiment_numeric"] = df["bert_sentiment"].replace({
    1: 1.0,
    0: 0.0,
    2: 0.5
})

movie_scores = (
    df.groupby("movieId")["bert_sentiment_numeric"]
    .mean()
    .reset_index()
)

movie_scores.columns = ["movieId", "bert_sentiment_score"]

movie_scores.to_csv(OUTPUT, index=False)
