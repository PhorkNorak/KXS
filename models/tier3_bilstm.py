"""Tier 3: BiLSTM + Self-Attention. Uses segmented text + word embeddings."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


def build_vocab(texts, max_vocab=30000):
    freq = {}
    for t in texts:
        for w in t.split():
            freq[w] = freq.get(w, 0) + 1
    w2i = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_vocab - 2]:
        w2i[w] = len(w2i)
    return w2i


class TextPairDataset(Dataset):
    def __init__(self, answers, references, scores, w2i, max_len=256):
        self.ans, self.ref, self.scores = answers, references, scores
        self.w2i, self.ml = w2i, max_len

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, idx):
        def tok(text):
            ids = [self.w2i.get(w, 1) for w in text.split()[:self.ml]]
            mask = [1] * len(ids)
            pad = self.ml - len(ids)
            ids += [0] * pad
            mask += [0] * pad
            return ids, mask

        a_ids, a_mask = tok(self.ans[idx])
        r_ids, r_mask = tok(self.ref[idx])
        return {
            "a_ids": torch.tensor(a_ids, dtype=torch.long),
            "a_mask": torch.tensor(a_mask, dtype=torch.float),
            "r_ids": torch.tensor(r_ids, dtype=torch.long),
            "r_mask": torch.tensor(r_mask, dtype=torch.float),
            "score": torch.tensor(self.scores[idx], dtype=torch.float),
        }


class BiLSTMAttention(nn.Module):
    name = "BiLSTM+Attention"

    def __init__(self, vocab_size, emb_dim=300, hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1))

    def _encode(self, ids, mask):
        out, _ = self.lstm(self.emb(ids))
        w = self.attn(out).squeeze(-1)
        w = w.masked_fill(~mask.bool(), -1e9)
        w = F.softmax(w, dim=-1)
        return torch.bmm(w.unsqueeze(1), out).squeeze(1)

    def forward(self, a_ids, a_mask, r_ids, r_mask, **kw):
        a = self._encode(a_ids, a_mask)
        r = self._encode(r_ids, r_mask)
        combined = torch.cat([a, r, a - r, a * r], dim=-1)
        return self.head(combined).squeeze(-1)


def train_bilstm(train_df, val_df, test_df, device, seed=42, epochs=30, patience=5):
    """Train BiLSTM+Attention and return test predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_texts = train_df["answer_seg"].tolist() + train_df["reference_seg"].tolist()
    w2i = build_vocab(all_texts)

    def make_loader(df, shuffle=False):
        ds = TextPairDataset(df["answer_seg"].tolist(), df["reference_seg"].tolist(),
                             df["score_ratio"].values, w2i)
        return DataLoader(ds, batch_size=16, shuffle=shuffle, num_workers=0)

    train_ld = make_loader(train_df, shuffle=True)
    val_ld = make_loader(val_df)
    test_ld = make_loader(test_df)

    model = BiLSTMAttention(len(w2i)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, wait, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for b in train_ld:
            b = {k: v.to(device) for k, v in b.items()}
            pred = model(b["a_ids"], b["a_mask"], b["r_ids"], b["r_mask"])
            loss = F.mse_loss(pred, b["score"])
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        vl = 0
        vn = 0
        with torch.no_grad():
            for b in val_ld:
                b = {k: v.to(device) for k, v in b.items()}
                pred = model(b["a_ids"], b["a_mask"], b["r_ids"], b["r_mask"])
                vl += F.mse_loss(pred, b["score"]).item()
                vn += 1

        avg_vl = vl / vn
        if ep % 5 == 0:
            print(f"    Epoch {ep}: val_loss={avg_vl:.4f}")
        if avg_vl < best_val:
            best_val = avg_vl
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    t_infer = time.perf_counter()
    with torch.no_grad():
        for b in test_ld:
            b = {k: v.to(device) for k, v in b.items()}
            p = model(b["a_ids"], b["a_mask"], b["r_ids"], b["r_mask"])
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(b["score"].cpu().numpy().tolist())
    infer_time_sec = round(time.perf_counter() - t_infer, 4)

    return np.clip(np.array(preds), 0, 1), np.array(labels), infer_time_sec
