"""Tier 4: Sentence-MiniLM + Cosine. Uses raw text."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SentenceMiniLMCosine:
    name = "S-MiniLM+Cosine"

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, *args, **kwargs):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        print(f"Loaded {self.model_name}")

    def predict(self, answers, references):
        if self.model is None:
            self.fit()
        a_emb = self.model.encode(answers, batch_size=32, show_progress_bar=False)
        r_emb = self.model.encode(references, batch_size=32, show_progress_bar=False)
        preds = [float(np.clip(cosine_similarity([a], [r])[0, 0], 0, 1))
                 for a, r in zip(a_emb, r_emb)]
        return np.array(preds)
