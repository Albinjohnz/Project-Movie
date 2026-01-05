import pandas as pd

INPUT = "./data/movies_with_bert_lstm.csv"
OUT = "./data/movies_with_combined_sentiment.csv"

W_BERT = 0.7
W_LSTM = 0.3

df = pd.read_csv(INPUT)

df["combined_sentiment_score"] = (
    W_BERT * df["bert_sentiment_score"] +
    W_LSTM * df["lstm_sentiment_score"]
)

df.to_csv(OUT, index=False)
print("Saved:", OUT)
print(df[["movieId","bert_sentiment_score","lstm_sentiment_score","combined_sentiment_score"]].head())
