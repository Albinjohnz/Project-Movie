import re
import json
import pandas as pd
import numpy as np
import torch
from torch import nn

INPUT = "./data/clean_reviews.csv"
MODEL_PATH = "./data/lstm_imdb_sentiment.pt"
WORD_INDEX_PATH = "./data/imdb_word_index.json"
OUTPUT = "./data/reviews_with_lstm_sentiment.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        idx = (lengths - 1).clamp(min=0)
        batch = torch.arange(out.size(0), device=out.device)
        last = out[batch, idx, :]
        x = torch.relu(self.fc1(last))
        x = self.drop(x)
        logits = self.fc2(x).squeeze(1)
        return logits

def load_word_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def text_to_imdb_sequence(text, word_index, vocab_size):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    index_from = 3
    seq = []
    for w in words:
        idx = word_index.get(w)
        if idx is None:
            seq.append(2)
        else:
            mapped = idx + index_from
            seq.append(2 if mapped >= vocab_size else mapped)
    return seq

def pad_sequences(seqs, max_len):
    X = np.zeros((len(seqs), max_len), dtype=np.int64)
    lengths = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)[:max_len]
        X[i, :len(s)] = s
        lengths[i] = max(1, len(s))
    return X, lengths

@torch.no_grad()
def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    vocab_size = ckpt["vocab_size"]
    max_len = ckpt["max_len"]
    emb_dim = ckpt["emb_dim"]
    hidden = ckpt["hidden"]

    word_index = load_word_index(WORD_INDEX_PATH)

    df = pd.read_csv(INPUT)
    texts = df["review"].astype(str).tolist()

    seqs = [text_to_imdb_sequence(t, word_index, vocab_size) for t in texts]
    X_np, len_np = pad_sequences(seqs, max_len)

    X = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    L = torch.tensor(len_np, dtype=torch.long, device=DEVICE)

    model = LSTMClassifier(vocab_size, emb_dim, hidden).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs_all = []
    batch_size = 256

    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        lb = L[i:i + batch_size]
        logits = model(xb, lb)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())

    probs_all = np.concatenate(probs_all)

    df["lstm_prob_positive"] = probs_all
    df["lstm_sentiment"] = (probs_all >= 0.5).astype(int)

    df.to_csv(OUTPUT, index=False)

if __name__ == "__main__":
    main()
