# src/detection/train_static_models.py
"""
===============================================================================
 Static Anomaly Model Training Script (Model Training Only)
===============================================================================

This script is responsible for:
1. Loading preprocessed and split datasets (train/validation/test).
2. Training multiple Isolation Forest models with different contamination levels.
3. Training a KMeans clustering model.
4. Saving all trained models to disk.

===============================================================================
 Outputs:
 - Trained models: *.pkl files (Isolation Forest, KMeans)
===============================================================================

Author : Ngueyep Ulrich
Date   : 2025-10-14
===============================================================================
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from config.config import Config
from src.detection.clustering_trainer import ModelTrainer


def main():
    """
    Main entry point for training static anomaly detection models.

    Steps:
    - Load preprocessed training data from disk.
    - Train Isolation Forest models with various contamination levels.
    - Train and save a KMeans clustering model.
    """
    config = Config()
    config.ensure_dirs()

    print("Starting static anomaly model training...")

    # === Step 1: Load preprocessed data ===
    x_train_path = os.path.join(config.PROCESSED_STATIC_DIR, "x_train.npy")
    y_train_path = os.path.join(config.PROCESSED_STATIC_DIR, "y_train.npy")

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # === Step 2: Train Models ===
    trainer = ModelTrainer(device="cpu", config=config)

    contamination_values = [0.01, 0.05, 0.10, 0.15]
    for contamination in contamination_values:
        print(f"Training Isolation Forest (contamination = {contamination})")
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_train)

        model_path = os.path.join(
            config.MODELS_DIR_STATIC_TRAIN,
            f"isolation_forest_{int(contamination * 100)}.pkl"
        )
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

    # === Step 3: Train KMeans ===
    print("Training KMeans model...")
    trainer.train_kmeans(X_train)

    print("All static models trained and saved successfully.")


if __name__ == "__main__":
    main()
