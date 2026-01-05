import pandas as pd

MOVIES_PATH = "./data/movie.csv"
BERT_PATH = "./data/movie_bert_sentiment_scores.csv"
LSTM_PATH = "./data/movie_lstm_sentiment_scores.csv"
OUT = "./data/movies_with_bert_lstm.csv"

movies = pd.read_csv(MOVIES_PATH)
bert = pd.read_csv(BERT_PATH)
lstm = pd.read_csv(LSTM_PATH)

df = movies.merge(bert, on="movieId", how="left").merge(lstm, on="movieId", how="left")

df["bert_sentiment_score"] = df["bert_sentiment_score"].fillna(0.5)
df["lstm_sentiment_score"] = df["lstm_sentiment_score"].fillna(0.5)

df.to_csv(OUT, index=False)
