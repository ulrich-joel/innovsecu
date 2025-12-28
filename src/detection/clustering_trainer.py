"""
===============================================================================
 Static Model Trainer Class
===============================================================================

This module defines the ModelTrainer class used for training unsupervised models 
such as KMeans on static malware datasets.

Features:
- Automatic or fixed KMeans training with silhouette score evaluation.
- Saving trained models to disk.

===============================================================================
 Outputs:
 - Trained KMeans model: kmeans.pkl
===============================================================================

Author : Ngueyep Ulrich
Date   : 2025-10-14
===============================================================================
"""

import logging
import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config.config import Config


class ModelTrainer:
    """
    Trainer class for unsupervised models like KMeans.

    Attributes:
        device (str): Computation device (default is 'cpu').
        config (Config): Configuration object containing paths.
    """
    def train_kmeans(self, X):
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(**self.config.CLUSTERING_PARAMS)
        self.kmeans.fit(X)
        print("✅ KMeans trained successfully.")

    def __init__(self, device="cpu", config=None):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.isolation_forest = None
        self.kmeans = None
        self.config = config or Config()

    def train_kmeans(self, X, auto_k=True, max_k=10, fixed_k=10):
        """
        Train a KMeans clustering model on the input data.

        If auto_k is True, automatically selects the best number of clusters
        (k) using the silhouette score, from 2 to max_k.

        If auto_k is False, uses the fixed number of clusters specified.

        Args:
            X (np.ndarray): Feature matrix.
            auto_k (bool): Whether to select k automatically using silhouette score.
            max_k (int): Maximum value of k to consider if auto_k is True.
            fixed_k (int): Fixed value of k to use if auto_k is False.

        Saves:
            Trained KMeans model to disk at config.MODELS_DIR_STATIC_TRAIN/kmeans.pkl
        """
        if auto_k:
            self.logger.info("🔍 Searching for the best number of clusters for KMeans...")
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
            self.logger.info(f"✅ Best k={best_k} with silhouette_score={best_score:.4f}")
        else:
            self.logger.info(f"📊 Training KMeans with fixed k={fixed_k}")
            self.kmeans = KMeans(n_clusters=fixed_k, random_state=42)
            self.kmeans.fit(X)

        # Save the trained KMeans model
        model_path = os.path.join(self.config.MODELS_DIR_STATIC_TRAIN, "kmeans.pkl")
        joblib.dump(self.kmeans, model_path)
        self.logger.info(f"💾 KMeans model saved to: {model_path}")
