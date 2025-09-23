import logging
import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config.config import Config

class ModelTrainer:
    def __init__(self, device="cpu", config=None):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.isolation_forest = None
        self.kmeans = None
        self.config = config or Config()

    def train_kmeans(self, X, auto_k=True, max_k=10, fixed_k=10):
        """
        Entraîne un modèle KMeans sur les données X.
        - Si auto_k=True, sélectionne automatiquement le meilleur nombre de clusters entre 2 et max_k (inclus)
        - Sinon, utilise le nombre de clusters fixé par fixed_k
        """
        if auto_k:
            self.logger.info("🔍 Recherche du meilleur nombre de clusters pour KMeans...")
            best_score = -1
            best_k = 2
            best_model = None

            for k in range(2, max_k + 1):
                model = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=42)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                self.logger.info(f"✔️ k={k} → silhouette_score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_model = model

            self.kmeans = best_model
            self.logger.info(f"✅ Meilleur k={best_k} avec silhouette_score={best_score:.4f}")
        else:
            self.logger.info(f"📊 Entraînement de KMeans avec k={fixed_k}")
            self.kmeans = KMeans(n_clusters=fixed_k, random_state=42)
            self.kmeans.fit(X)

        # Sauvegarde du modèle
        model_path = os.path.join(self.config.MODELS_DIR, "kmeans.pkl")
        joblib.dump(self.kmeans, model_path)
        self.logger.info(f"💾 KMeans sauvegardé dans : {model_path}")
