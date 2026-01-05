import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

INPUT = "./data/clean_reviews.csv"
OUTPUT = "./data/reviews_with_bert_sentiment.csv"

# You can choose a pre-trained sentiment model.
# Example: 5-star sentiment model
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Loading data...")
df = pd.read_csv(INPUT)

def bert_predict_sentiment(text: str) -> int:
    """
    Use BERT to get a sentiment score.
    For this model:
    1 star = very negative
    2 star = negative
    3 star = neutral
    4 star = positive
    5 star = very positive

    We map to:
    0 = negative
    1 = positive
    2 = neutral
    """

    if not isinstance(text, str) or text.strip() == "":
        return 2  # neutral for empty

    encoded = tokenizer(import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

INPUT = "./data/clean_reviews.csv"
OUTPUT = "./data/reviews_with_bert_sentiment.csv"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Loading data...")
df = pd.read_csv(INPUT)

def bert_predict_sentiment(text: str) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 2

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    stars = pred + 1

    if stars <= 2:
        return 0
    elif stars == 3:
        return 2
    else:
        return 1

print("Computing BERT sentiment on reviews...")
sentiments = []
for text in tqdm(df["review"], total=len(df)):
    sentiments.append(bert_predict_sentiment(text))

df["bert_sentiment"] = sentiments

print("Saving:", OUTPUT)
df.to_csv(OUTPUT, index=False)
print("Done. Label distribution:")
print(df["bert_sentiment"].value_counts())

        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()  # 0..4

    stars = pred + 1  # 1..5

    if stars <= 2:
        return 0  # negative
    elif stars == 3:
        return 2  # neutral
    else:
        return 1  # positive


print("Computing BERT sentiment on reviews...")
sentiments = []
for text in tqdm(df["review"], total=len(df)):
    sentiments.append(bert_predict_sentiment(text))

df["bert_sentiment"] = sentiments

print("Saving:", OUTPUT)
df.to_csv(OUTPUT, index=False)
print("Done. Label distribution:")
print(df["bert_sentiment"].value_counts())
