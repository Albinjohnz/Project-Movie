import pandas as pd
import requests
import time
import os

# ---- PATHS ----
LINKS_PATH = "./data/links.csv"
OUT_PATH = "./data/tmdb_reviews_full.csv"

TMDB_API_KEY = "a5759fc715395f6dd01ae20464279881"

BASE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"


def fetch_reviews_for_tmdb_id(tmdb_id: int):
    """Fetch ALL review pages for a single TMDB movie id."""
    page = 1
    collected = []

    while True:
        url = BASE_URL.format(tmdb_id=tmdb_id)
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page,
        }

        response = requests.get(url, params=params)

        # Handle rate limiting politely
        if response.status_code == 429:
            print("Rate limit reached, waiting a bit...")
            time.sleep(10)
            continue

        if response.status_code != 200:
            # Non-fatal: just skip this movie
            print(f"TMDB error {response.status_code} for tmdbId={tmdb_id}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for r in results:
            collected.append(
                {
                    "author": r.get("author"),
                    "review": r.get("content"),
                    "created_at": r.get("created_at"),
                }
            )

        total_pages = data.get("total_pages", 1)
        if page >= total_pages:
            break

        page += 1
        time.sleep(0.25)  # small delay between pages

    return collected


def main():
    links = pd.read_csv(LINKS_PATH)
    links = links.dropna(subset=["tmdbId"])

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done_tmdb_ids = set(existing["tmdbId"].unique())
        all_rows = existing.to_dict("records")
        print(f"Resuming: already have reviews for {len(done_tmdb_ids)} TMDB IDs")
    else:
        done_tmdb_ids = set()
        all_rows = []

    total = len(links)

    for idx, row in links.iterrows():
        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])

        if tmdb_id in done_tmdb_ids:
            continue

        print(f"[{idx+1}/{total}] movieId={movie_id}, tmdbId={tmdb_id}")

        reviews = fetch_reviews_for_tmdb_id(tmdb_id)

        for rev in reviews:
            all_rows.append(
                {
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    "author": rev["author"],
                    "review": rev["review"],
                    "created_at": rev["created_at"],
                }
            )

        # Save progress after each movieId
        pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)

        # polite delay between movies
        time.sleep(0.4)

    print("Done fetching reviews.")
    print(f"Total reviews collected: {len(all_rows)}")


if __name__ == "__main__":
    main()
