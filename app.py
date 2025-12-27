import streamlit as st
import pandas as pd
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
MOVIES_PATH = "./data/movies_with_sentiment.csv"
TAGS_PATH   = "./data/tag.csv"
REVIEWS_PATH = "./data/reviews_with_sentiment.csv"  # reviews + sentiment labels


def normalize_title(s: str) -> str:
    """
    Lowercase, remove all non-alphanumeric characters.
    'Toy Story (1995)' -> 'toystory1995'
    'toy story' -> 'toystory'
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@st.cache_resource
def load_data_and_model():
    # Load movies, tags, and reviews
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    # Group tags per movie
    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(str(t).replace(" ", "_") for t in x))
        .reset_index()
    )

    # Merge movies with tags
    movies = movies.merge(tags_grouped, on="movieId", how="left")

    # Clean NaNs
    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["sentiment_score"] = movies["sentiment_score"].fillna(0.5)

    # Combined features for content-based similarity
    movies["combined_features"] = (
        movies["genres"].str.replace("|", " ") + " " + movies["tag"]
    ).str.strip()

    movies["title_clean"] = movies["title"].astype(str).apply(normalize_title)

    # TF-IDF + cosine similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, cosine_sim, reviews


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

    # Exact-ish match first
    if not matches.empty:
        idx = matches.index[0]
        return idx, matches.iloc[0]["title"]

    # Fuzzy match with cleaned title
    all_clean = movies["title_clean"].tolist()
    best_matches = difflib.get_close_matches(key_clean, all_clean, n=1, cutoff=0.5)

    if not best_matches:
        return None, None

    best_clean = best_matches[0]
    idx = movies.index[movies["title_clean"] == best_clean][0]
    return idx, movies.loc[idx, "title"]


def recommend(title, movies, cosine_sim, top_k=10, w_sim=0.7, w_sent=0.3):
    """
    Sentiment-aware recommendation:
    final_score = w_sim * similarity + w_sent * sentiment_score
    """
    movies = movies.reset_index(drop=True)
    idx, matched_title = find_movie_index(title, movies)
    if idx is None:
        return matched_title, pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_k + 1]

    movie_indices = [i for i, score in sim_scores]
    sim_values = [score for i, score in sim_scores]

    recs = movies.iloc[movie_indices][
        ["movieId", "title", "genres", "sentiment_score"]
    ].copy()
    recs["similarity"] = sim_values
    recs["final_score"] = (
        w_sim * recs["similarity"] + w_sent * recs["sentiment_score"]
    )

    recs = recs.sort_values(by="final_score", ascending=False)
    return matched_title, recs


#UI #

st.title("Sentiment-Aware Movie Recommender")

st.write(
    "Type a movie you like and get recommendations that are similar **and** well-liked by audiences.\n"
    "Below the match, you'll also see real audience reviews with sentiment labels."
)

movies, cosine_sim, reviews = load_data_and_model()

user_title = st.text_input("Enter a movie title:", "Toy Story")

col1, col2 = st.columns(2)
with col1:
    w_sim = st.slider("Weight for similarity", 0.0, 1.0, 0.7, 0.05)
with col2:
    w_sent = st.slider("Weight for sentiment", 0.0, 1.0, 0.3, 0.05)

if st.button("Recommend") and user_title.strip():
    matched_title, recs = recommend(
        user_title,
        movies,
        cosine_sim,
        top_k=15,
        w_sim=w_sim,
        w_sent=w_sent,
    )

    if matched_title is None or recs.empty:
        st.warning(f'No movie found similar to "{user_title}". Try another name.')
    else:
        st.success(f'Best match: **{matched_title}**')

        # Reviews for the matched movie -
        st.write("### 📝 Audience Reviews")

        matched_movie = movies[movies["title"] == matched_title]

        if not matched_movie.empty:
            mid = int(matched_movie.iloc[0]["movieId"])
            movie_reviews = reviews[reviews["movieId"] == mid]

            if movie_reviews.empty:
                st.info("No reviews available for this movie.")
            else:
                # Map numeric sentiment to label
                sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}

                for _, r in movie_reviews.head(6).iterrows():
                    raw_sent = r.get("sentiment", 2)
                    try:
                        sent_int = int(raw_sent)
                    except Exception:
                        sent_int = 2
                    sent_label = sentiment_map.get(sent_int, "Neutral")

                    author = r["author"] if pd.notna(r["author"]) else "Anonymous"

                    st.markdown(
                        f"""
**Author:** *{author}*  
**Sentiment:** **{sent_label}**

> {r['review']}

---
                        """
                    )
        else:
            st.info("Movie not found for review lookup.")

        #  Recommended movies table 
        st.write("### 🎯 Recommended Movies")
        st.dataframe(
            recs[["title", "genres", "similarity", "sentiment_score", "final_score"]]
            .reset_index(drop=True)
        )
