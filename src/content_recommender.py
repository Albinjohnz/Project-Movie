# src/content_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
MOVIES_PATH = "./data/movie.csv"
TAGS_PATH = "./data/tag.csv"

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

    movies["combined_features"] = (
        movies["genres"].str.replace("|", " ") + " " + movies["tag"]
    ).str.strip()

    return movies

def build_similarity_matrix(movies: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title: str, movies: pd.DataFrame, cosine_sim):
    movies_lower = movies["title"].str.lower()
    key = title.strip().lower()

    # Find any movie containing the search text
    matches = movies[movies_lower.str.contains(key, na=False)]

    if matches.empty:
        print(f'\n No movie found containing "{title}". Try another name.')
        return pd.DataFrame()

    # Pick first match
    idx = matches.index[0]
    print(f'\n Best match found: {matches.iloc[0]["title"]}')

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i for i, score in sim_scores]
    scores = [score for i, score in sim_scores]

    recs = movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
    recs["similarity"] = scores
    return recs


if __name__ == "__main__":
    print("Loading data...")
    movies = load_data()
    print("Number of movies:", len(movies))

    print("Building similarity matrix...")
    cosine_sim = build_similarity_matrix(movies)

    while True:
        title = input("\n Enter a movie title (or 'q' to quit): ").strip()
        if title.lower() == "q":
            break

        recs = get_recommendations(title, movies, cosine_sim)

        if not recs.empty:
            print("\nTop similar movies:\n")
            print(recs.to_string(index=False))
