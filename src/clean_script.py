import pandas as pd
import re

INPUT_PATH = "./data/tmdb_reviews_full.csv"
OUTPUT_PATH = "./data/clean_reviews.csv"

def is_english(text: str, threshold: float = 0.7) -> bool:
    """
    Returns True if the text looks mostly English (based on A–Z letters ratio).
    threshold = 0.7 means at least 70% of characters are A–Z.
    """
    if not isinstance(text, str):
        return False

    if text.strip() == "":
        return False

    letters = re.findall(r"[A-Za-z]", text)
    ratio = len(letters) / max(len(text), 1)
    return ratio >= threshold

def main():
    print("Loading reviews from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    print("Total rows at start:", len(df))

    # 1. Drop rows with missing review text
    df = df.dropna(subset=["review"])
    print("After dropping NaN reviews:", len(df))

    # 2. Remove exact duplicate review texts
    df = df.drop_duplicates(subset=["review"])
    print("After removing duplicate reviews:", len(df))

    # 3. Keep only English-like reviews
    df = df[df["review"].apply(is_english)]
    print("After keeping English-like text only:", len(df))

    # 4. Save cleaned file
    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved cleaned reviews to:", OUTPUT_PATH)

    # Show a small sample to inspect
    print("\nSample cleaned rows:")
    print(df.head(5))

if __name__ == "__main__":
    main()
