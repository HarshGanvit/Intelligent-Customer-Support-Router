"""Model loading + inference for the Streamlit app (inference only, no training code)."""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, DistilBertModel

from routing import DEFAULT_CONFIG, best_match


class MultiTaskSupportRouter(nn.Module):
    """Identical layer names to the training notebook, so the saved state_dict loads without renaming."""

    def __init__(self, config, n_dept, n_urg, n_sent):
        super().__init__()
        self.distilbert = DistilBertModel(config)
        h = self.distilbert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.department_classifier = nn.Linear(h, n_dept)
        self.sentiment_classifier = nn.Linear(h, n_sent)
        self.urgency_classifier = nn.Linear(h, n_urg)

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.department_classifier(pooled), self.sentiment_classifier(pooled), self.urgency_classifier(pooled)


class RouterEngine:
    def __init__(self, art_dir):
        self.art_dir = art_dir
        with open(os.path.join(art_dir, "labels.json")) as f:
            labels = json.load(f)
        self.departments = labels["departments"]
        self.urgencies = labels["urgencies"]
        self.sentiments = labels["sentiments"]
        self.max_len = int(labels.get("max_len", 64))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # classifier (config + tokenizer are saved next to the weights, nothing is downloaded)
        self.tokenizer = AutoTokenizer.from_pretrained(art_dir)
        config = AutoConfig.from_pretrained(art_dir)
        self.model = MultiTaskSupportRouter(config, len(self.departments), len(self.urgencies), len(self.sentiments))
        state = torch.load(os.path.join(art_dir, "router_weights.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # retrieval encoder: use the local copy if the notebook saved it, else download from the Hub
        local_emb = os.path.join(art_dir, "retrieval_model")
        emb_src = local_emb if os.path.isdir(local_emb) else labels["retrieval_model"]
        self.emb_tok = AutoTokenizer.from_pretrained(emb_src)
        self.emb_model = AutoModel.from_pretrained(emb_src).to(self.device).eval()

        self.pool_embs = np.load(os.path.join(art_dir, "pool_embeddings.npy"))
        self.replies = pd.read_csv(os.path.join(art_dir, "replies.csv"))
        self.pool_cats = self.replies["category"].values

        self.cfg = dict(DEFAULT_CONFIG)
        cfg_path = os.path.join(art_dir, "routing_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.cfg.update(json.load(f))

    @torch.no_grad()
    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len,
                             return_tensors="pt").to(self.device)
        d, s, u = self.model(enc["input_ids"], enc["attention_mask"])

        def head(logits, classes):
            p = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs = {str(c): round(float(x), 4) for c, x in zip(classes, p)}
            top = max(probs, key=probs.get)
            return top, probs[top], probs

        dept, dept_conf, dept_probs = head(d, self.departments)
        sent, sent_conf, sent_probs = head(s, self.sentiments)
        urg, urg_conf, urg_probs = head(u, self.urgencies)
        return {"text": text, "department": dept, "department_confidence": dept_conf, "department_probs": dept_probs,
                "sentiment": sent, "sentiment_confidence": sent_conf, "sentiment_probs": sent_probs,
                "urgency": urg, "urgency_confidence": urg_conf, "urgency_probs": urg_probs}

    @torch.no_grad()
    def embed(self, texts, max_len=64):
        enc = self.emb_tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        h = self.emb_model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(e, p=2, dim=1).cpu().numpy().astype("float32")

    def retrieve(self, text, dept, dept_conf, cfg=None):
        q = self.embed([text])[0]
        i, sim = best_match(q, self.pool_embs, self.pool_cats, dept, dept_conf, cfg or self.cfg)
        row = self.replies.iloc[i]
        return {"reply": str(row["response"]), "similarity": sim, "matched_instruction": str(row["instruction"])}
