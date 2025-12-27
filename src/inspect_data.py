import pandas as pd

MOVIES_PATH = r"D:\DBS\Resarch Paper\code\data\movie.csv"
TAGS_PATH = r"D:\DBS\Resarch Paper\code\data\tag.csv"
LINKS_PATH = r"D:\DBS\Resarch Paper\code\data\links.csv"

def show_basic_info():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)
    links = pd.read_csv(LINKS_PATH)

    print("Movies head:")
    print(movies.head())

    print("\nTags head:")
    print(tags.head())

    print("\nLinks head:")
    print(links.head())

    print("\nColumns in links.csv:")
    print(links.columns)

if __name__ == "__main__":
    show_basic_info()
