import pandas as pd

MOVIES_PATH = r"D:\DBS\Resarch Paper\code\data\movie.csv"
TAGS_PATH = r"D:\DBS\Resarch Paper\code\data\tag.csv"

def show_basic_info():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)

    print("Movies head:")
    print(movies.head())

    print("Tags head:")
    print(tags.head())

if __name__ == "__main__":
    show_basic_info()
