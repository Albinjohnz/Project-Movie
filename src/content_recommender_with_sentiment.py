import pandas as pd
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use movies WITH sentiment
MOVIES_PATH = "./data/movies_with_sentiment.csv"
TAGS_PATH   = "./data/tag.csv"   


def normalize_title(s: str) -> str:
    """
    Lowercase, remove all non-alphanumeric characters.
    'Toy Story (1995)' -> 'toystory1995'
    'toy story' -> 'toystory'
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)

    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(str(t).replace(" ", "_") for t in x))
        .reset_index()
    )

    movies = movies.merge(tags_grouped, on="movieId", how="left")

    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    if "sentiment_score" in movies.columns:
        movies["sentiment_score"] = movies["sentiment_score"].fillna(0.5)
    else:
        movies["sentiment_score"] = 0.5

    movies["combined_features"] = (
        movies["genres"].str.replace("|", " ") + " " + movies["tag"]
    ).str.strip()

    movies["title_clean"] = movies["title"].astype(str).apply(normalize_title)

    return movies


def build_similarity_matrix(movies: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def find_movie_index(user_title: str, movies: pd.DataFrame):
    """
    1) Try normal 'contains' search (case-insensitive).
    2) If no match, fuzzy match using cleaned titles.
    Returns (index, matched_title) or (None, None).
    """
    key_raw = user_title.strip().lower()
    key_clean = normalize_title(user_title)

    titles_lower = movies["title"].str.lower()

    matches = movies[titles_lower.str.contains(key_raw, na=False)]
    if not matches.empty:
        idx = matches.index[0]
        return idx, matches.iloc[0]["title"]

    all_clean = movies["title_clean"].tolist()
    best_matches = difflib.get_close_matches(key_clean, all_clean, n=1, cutoff=0.5)

    if not best_matches:
        return None, None

    best_clean = best_matches[0]
    idx = movies.index[movies["title_clean"] == best_clean][0]
    return idx, movies.loc[idx, "title"]


def get_recommendations_with_sentiment(
    title: str,
    movies: pd.DataFrame,
    cosine_sim,
    w_sim: float = 0.7,
    w_sent: float = 0.3,
):
    movies = movies.reset_index(drop=True)

    idx, matched_title = find_movie_index(title, movies)
    if idx is None:
        print(f'\n No movie found similar to "{title}". Try another name.')
        return pd.DataFrame()

    print(f'\n Best match found: {matched_title}')

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]  # top 20

    movie_indices = [i for i, score in sim_scores]
    sim_values = [score for i, score in sim_scores]

    recs = movies.iloc[movie_indices][["movieId", "title", "genres", "sentiment_score"]].copy()
    recs["similarity"] = sim_values

    recs["final_score"] = w_sim * recs["similarity"] + w_sent * recs["sentiment_score"]

    recs = recs.sort_values(by="final_score", ascending=False)

    recs = recs[["movieId", "title", "genres", "similarity", "sentiment_score", "final_score"]]

    return recs


if __name__ == "__main__":
    print("Loading data...")
    movies = load_data()
    print("Number of movies:", len(movies))

    print("Building similarity matrix (this may take a bit)...")
    cosine_sim = build_similarity_matrix(movies)

    while True:
        title = input("\n🎬 Enter a movie title (or 'q' to quit): ").strip()
        if title.lower() == "q":
            break

        recs = get_recommendations_with_sentiment(title, movies, cosine_sim)

        if not recs.empty:
            print("\nTop sentiment-aware similar movies:\n")
            print(recs.to_string(index=False))
