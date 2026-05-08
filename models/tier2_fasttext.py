"""Tier 2: FastText + Cosine Similarity. Uses segmented text."""
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FastTextCosine:
    name = "FastText+Cosine"

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path

    def fit(self, train_ans=None, train_ref=None, train_scores=None):
        try:
            import fasttext
            import fasttext.util
            if self.model_path and os.path.exists(self.model_path):
                self.model = fasttext.load_model(self.model_path)
            else:
                print("Downloading FastText Khmer model (cc.km.300.bin)...")
                fasttext.util.download_model("km", if_exists="ignore")
                self.model = fasttext.load_model("cc.km.300.bin")
            print(f"FastText loaded: dim={self.model.get_dimension()}")
        except ImportError:
            print("ERROR: pip install fasttext")
        except Exception as e:
            print(f"FastText error: {e}")

    def predict(self, answers, references):
        if self.model is None:
            print("FastText not loaded, returning 0.5")
            return np.full(len(answers), 0.5)
        preds = []
        for a, r in zip(answers, references):
            va = self.model.get_sentence_vector(a)
            vr = self.model.get_sentence_vector(r)
            if np.linalg.norm(va) == 0 or np.linalg.norm(vr) == 0:
                preds.append(0.5)
            else:
                sim = cosine_similarity([va], [vr])[0, 0]
                preds.append(float(np.clip(sim, 0, 1)))
        return np.array(preds)
