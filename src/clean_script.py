import pandas as pd
import re

INPUT_PATH = "./data/tmdb_reviews_full.csv"
OUTPUT_PATH = "./data/clean_reviews.csv"

def is_english(text, threshold=0.7):
    if not isinstance(text, str):
        return False
    if text.strip() == "":
        return False
    letters = re.findall(r"[A-Za-z]", text)
    ratio = len(letters) / max(len(text), 1)
    return ratio >= threshold

def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["review"])
    df = df.drop_duplicates(subset=["review"])
    df = df[df["review"].apply(is_english)]
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
