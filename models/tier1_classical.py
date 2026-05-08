"""
Tier 1: BoW+Cosine, TF-IDF+Cosine (no training)
Tier 2: TF-IDF+SVR (trained)
All use SEGMENTED text.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity


class BoWCosine:
    name = "BoW+Cosine"

    def __init__(self):
        self.vec = CountVectorizer()

    def fit(self, train_ans, train_ref, train_scores=None):
        self.vec.fit(train_ans + train_ref)

    def predict(self, answers, references):
        preds = []
        for a, r in zip(answers, references):
            try:
                v = self.vec.transform([a, r])
                sim = cosine_similarity(v[0:1], v[1:2])[0, 0]
                preds.append(float(np.clip(sim, 0, 1)))
            except Exception:
                preds.append(0.5)
        return np.array(preds)


class TFIDFCosine:
    name = "TF-IDF+Cosine"

    def __init__(self, max_features=10000):
        self.vec = TfidfVectorizer(max_features=max_features)

    def fit(self, train_ans, train_ref, train_scores=None):
        self.vec.fit(train_ans + train_ref)

    def predict(self, answers, references):
        preds = []
        for a, r in zip(answers, references):
            try:
                v = self.vec.transform([a, r])
                sim = cosine_similarity(v[0:1], v[1:2])[0, 0]
                preds.append(float(np.clip(sim, 0, 1)))
            except Exception:
                preds.append(0.5)
        return np.array(preds)


class TFIDFSVR:
    name = "TF-IDF+SVR"

    def __init__(self, max_features=10000):
        self.vec = TfidfVectorizer(max_features=max_features)
        self.svr = SVR(kernel="rbf", C=1.0, epsilon=0.05)

    def fit(self, train_ans, train_ref, train_scores):
        self.vec.fit(train_ans + train_ref)
        feats = self._features(train_ans, train_ref)
        self.svr.fit(feats, train_scores)

    def predict(self, answers, references):
        feats = self._features(answers, references)
        return np.clip(self.svr.predict(feats), 0, 1)

    def _features(self, answers, references):
        feats = []
        for a, r in zip(answers, references):
            va = self.vec.transform([a]).toarray()[0]
            vr = self.vec.transform([r]).toarray()[0]
            sim = cosine_similarity([va], [vr])[0, 0]
            feats.append([
                sim,                          # cosine similarity
                np.linalg.norm(va),           # answer magnitude
                np.linalg.norm(vr),           # reference magnitude
                np.linalg.norm(va - vr),      # difference magnitude
                len(a.split()),               # answer word count
                len(r.split()),               # reference word count
                len(a.split()) / max(len(r.split()), 1),  # length ratio
            ])
        return np.array(feats)
