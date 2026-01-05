import os
import urllib.request
import random
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")

OUT_MODEL = "./data/lstm_imdb_sentiment.pt"

VOCAB_SIZE = 20000
MAX_LEN = 200
EMB_DIM = 128
HIDDEN = 128

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
SEED = 42
PATIENCE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMDB_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
IMDB_NPZ_PATH = "./data/imdb.npz"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def download_if_needed(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading IMDB dataset to {path} ...")
        urllib.request.urlretrieve(url, path)

def clamp_vocab(seqs, vocab_size):
    out = []
    for s in seqs:
        out.append([w if w < vocab_size else 2 for w in s])
    return out

def pad_sequences_np(seqs, maxlen, padding_value=0):
    out = np.full((len(seqs), maxlen), padding_value, dtype=np.int64)
    lengths = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)[:maxlen]
        out[i, :len(s)] = s
        lengths[i] = len(s)
    lengths = np.clip(lengths, 1, maxlen)
    return out, lengths

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

def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return (preds == y).float().mean().item()

@torch.no_grad()
def eval_loader(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    for xb, lb, yb in loader:
        xb = xb.to(DEVICE)
        lb = lb.to(DEVICE)
        yb_f = yb.to(DEVICE).float()
        logits = model(xb, lb)
        loss = loss_fn(logits, yb_f)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, yb.to(DEVICE)) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n

def main():
    set_seed(SEED)

    download_if_needed(IMDB_NPZ_URL, IMDB_NPZ_PATH)

    with np.load(IMDB_NPZ_PATH, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    x_train = clamp_vocab(x_train, VOCAB_SIZE)
    x_test = clamp_vocab(x_test, VOCAB_SIZE)

    x_train_pad, len_train = pad_sequences_np(x_train, MAX_LEN)
    x_test_pad, len_test = pad_sequences_np(x_test, MAX_LEN)

    x_train_t = torch.tensor(x_train_pad, dtype=torch.long)
    len_train_t = torch.tensor(len_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    x_test_t = torch.tensor(x_test_pad, dtype=torch.long)
    len_test_t = torch.tensor(len_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    n = x_train_t.size(0)
    n_val = int(0.2 * n)

    x_val_t = x_train_t[:n_val]
    len_val_t = len_train_t[:n_val]
    y_val_t = y_train_t[:n_val]

    x_tr_t = x_train_t[n_val:]
    len_tr_t = len_train_t[n_val:]
    y_tr_t = y_train_t[n_val:]

    train_loader = DataLoader(TensorDataset(x_tr_t, len_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, len_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_t, len_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(VOCAB_SIZE, EMB_DIM, HIDDEN).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, lb, yb in train_loader:
            xb = xb.to(DEVICE)
            lb = lb.to(DEVICE)
            yb_f = yb.to(DEVICE).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = loss_fn(logits, yb_f)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        train_loss = total_loss / total_n
        val_loss, val_acc = eval_loader(model, val_loader, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping: val_loss did not improve for {PATIENCE} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = eval_loader(model, test_loader, loss_fn)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "max_len": MAX_LEN,
            "emb_dim": EMB_DIM,
            "hidden": HIDDEN,
        },
        OUT_MODEL,
    )
    print("Done.")

if __name__ == "__main__":
    main()
