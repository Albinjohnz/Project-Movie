import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

INPUT = "./data/clean_reviews.csv"
OUTPUT = "./data/reviews_with_sentiment.csv"

print("Loading data...")
df = pd.read_csv(INPUT)

sia = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]

    if score > 0.05:
        return 1      # Positive
    elif score < -0.05:
        return 0      # Negative
    else:
        return 2      # Neutral

print("Computing sentiment…")
df["sentiment"] = df["review"].apply(label_sentiment)

print("Saving...")
df.to_csv(OUTPUT, index=False)

print("Done!")
print(df["sentiment"].value_counts())
