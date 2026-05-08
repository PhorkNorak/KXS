"""
Tier 5: Transformer Fine-tuning
Simple: [CLS] answer [SEP] reference [SEP] → Encoder → [CLS] → MLP → score
Dual:   Encode(answer)→A, Encode(reference)→R → [A;R;A-R;A*R] → MLP → score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TransformerSimple(nn.Module):
    def __init__(self, model_name, dropout=0.2, freeze_layers=6):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hs = self.encoder.config.hidden_size
        self._freeze(freeze_layers)
        self.head = nn.Sequential(
            nn.Linear(hs, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))
        self.name = f"Simple({model_name.split('/')[-1]})"

    def _freeze(self, n):
        if n <= 0:
            return
        if hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "layer"):
            layers = self.encoder.layer
        if layers:
            for i in range(min(n, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kw):
        args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "bert-base" in self.model_name and "roberta" not in self.model_name:
            args["token_type_ids"] = token_type_ids
        out = self.encoder(**args)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)


class TransformerDual(nn.Module):
    def __init__(self, model_name, dropout=0.2, freeze_layers=6):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hs = self.encoder.config.hidden_size
        self._freeze(freeze_layers)
        self.head = nn.Sequential(
            nn.Linear(hs * 4, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 1))
        self.name = f"Dual({model_name.split('/')[-1]})"

    def _freeze(self, n):
        if n <= 0:
            return
        if hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "layer"):
            layers = self.encoder.layer
        if layers:
            for i in range(min(n, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False

    def _enc(self, ids, mask, ttids=None):
        args = {"input_ids": ids, "attention_mask": mask}
        if ttids is not None and "bert-base" in self.model_name and "roberta" not in self.model_name:
            args["token_type_ids"] = ttids
        return self.encoder(**args).last_hidden_state[:, 0, :]

    def forward(self, ans_input_ids, ans_attention_mask, ref_input_ids, ref_attention_mask,
                ans_token_type_ids=None, ref_token_type_ids=None, **kw):
        a = self._enc(ans_input_ids, ans_attention_mask, ans_token_type_ids)
        r = self._enc(ref_input_ids, ref_attention_mask, ref_token_type_ids)
        combined = torch.cat([a, r, a - r, a * r], dim=-1)
        return self.head(combined).squeeze(-1)


def train_transformer(model, train_ld, val_ld, device, lr=2e-5, epochs=30,
                      patience=5, wd=0.01, warmup=0.1):
    """Train transformer with early stopping. Returns best model."""
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    total = len(train_ld) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup), total)

    best_vl, wait, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        model.train()
        tl, tn = 0, 0
        for b in train_ld:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            score = b.pop("score")
            pred = model(**b)
            loss = F.mse_loss(pred, score)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tl += loss.item()
            tn += 1

        model.eval()
        vl, vn = 0, 0
        with torch.no_grad():
            for b in val_ld:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                score = b.pop("score")
                pred = model(**b)
                vl += F.mse_loss(pred, score).item()
                vn += 1

        avg_vl = vl / vn
        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep}: train={tl/tn:.4f} val={avg_vl:.4f}")
        if avg_vl < best_vl:
            best_vl = avg_vl
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop ep {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_transformer(model, loader, device):
    model.eval()
    preds, labels = [], []
    for b in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        score = b.pop("score")
        pred = model(**b)
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(score.cpu().numpy().tolist())
    return np.clip(np.array(preds), 0, 1), np.array(labels)
