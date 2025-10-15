"""
train_dynamic_models.py
=============================================================================================
 Dynamic Train Model Script
=============================================================================================

This script is responsible for training dynamic models for anomaly detection and clustering.
It includes the following steps:
1. Load and normalize dynamic data.
2. Train a Variational Autoencoder (VAE) for anomaly detection.
3. Train an Isolation Forest for unsupervised anomaly detection.
4. Train a KMeans model for clustering.
5. Evaluate the models and save the results.

Dependencies:
- TensorFlow for the VAE model.
- Scikit-learn for Isolation Forest and KMeans.
- Pandas and NumPy for data manipulation.
- Joblib for saving and loading models.
- Config module for project-specific configurations.

=============================================================================================

Outputs:
- Trained models saved in the `CHECKPOINT_DIR`.
- Evaluation report saved as a CSV file in the `REPORT_FILE`.

Author: Ngueyep Ulrich
=============================================================================================
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from config.config import Config

# === Initial Configuration ===
cfg = Config()
cfg.ensure_dirs()  # Ensure necessary directories exist
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_trainer")

# Define paths for saving models and reports
CHECKPOINT_DIR = cfg.MODELS_DIR
REPORT_FILE = os.path.join(cfg.REPORTS_DIR, "dynamic_model_report.csv")
os.makedirs(cfg.REPORTS_DIR, exist_ok=True)

# Define the features and label column
features = ['processId', 'threadId', 'parentProcessId', 'userId', 'eventId', 'argsNum', 'returnValue', 'behavior_cluster', 'stack_depth']
label_col = 'sus'

def load_data():
    """
    Load training, validation, and testing datasets.

    Returns:
        tuple: DataFrames for training, validation, and testing.
    """
    logger.info("Loading data files...")
    train = pd.read_csv(cfg.DYNAMIC_FILE)
    val = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, "labelled_validation_data.csv"))
    test = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, "labelled_testing_data.csv"))
    return train, val, test

def normalize(train, val, test):
    """
    Normalize the datasets using StandardScaler.

    Args:
        train (pd.DataFrame): Training dataset.
        val (pd.DataFrame): Validation dataset.
        test (pd.DataFrame): Testing dataset.

    Returns:
        tuple: Normalized datasets (X_train, X_val, X_test) and their labels.
    """
    logger.info("Normalizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    return X_train, X_val, X_test, train[label_col], val[label_col], test[label_col]

class BetaScheduler(tf.keras.callbacks.Callback):
    """
    Custom callback to schedule the beta parameter for the VAE during training.

    Attributes:
        max_beta (float): Maximum value of beta.
        total_epochs (int): Total number of epochs.
        beta (float): Current beta value.
    """
    def __init__(self, max_beta, total_epochs):
        self.max_beta = max_beta
        self.total_epochs = total_epochs
        self.beta = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Update beta at the end of each epoch.
        """
        self.beta = min(self.max_beta, (epoch + 1) / self.total_epochs * self.max_beta)

def build_vae(input_dim, encoding_dim=4):
    # Encoder
    encoder_inputs = Input(shape=(input_dim,), name="encoder_input")
    h = Dense(16, activation='relu')(encoder_inputs)
    z_mean = Dense(encoding_dim, name="z_mean")(h)
    z_log_var = Dense(encoding_dim, name="z_log_var")(h)

    # Fonction de sampling pour la variabilité
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], encoding_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, name="z")([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(16, activation='relu')
    decoder_out = Dense(input_dim, activation='linear')

    class VAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Définir correctement les sous-modèles
            self.encoder_inputs = encoder_inputs
            self.encoder = tf.keras.Model(encoder_inputs, z)  # Définir 'encoder' ici
            self.z_mean_layer = tf.keras.Model(encoder_inputs, z_mean)
            self.z_log_var_layer = tf.keras.Model(encoder_inputs, z_log_var)
            self.decoder_h = decoder_h
            self.decoder_out = decoder_out

        def call(self, inputs):
            z = self.encoder(inputs)  # Encoder doit être défini ici
            h_decoded = self.decoder_h(z)
            outputs = self.decoder_out(h_decoded)
            return outputs

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]

            with tf.GradientTape() as tape:
                reconstructed = self(data)
                z_mean = self.z_mean_layer(data)
                z_log_var = self.z_log_var_layer(data)

                # Calcul des pertes
                reconstruction_loss = tf.reduce_mean(tf.square(data - reconstructed), axis=1)
                kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": total_loss}

    vae = VAE()
    #vae.compile(optimizer='adam', run_eagerly=True)
    vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)

    return vae


def train_models():
    """
    Train dynamic models (VAE, Isolation Forest, KMeans) and evaluate their performance.
    """
    # Load and normalize data
    train_df, val_df, test_df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = normalize(train_df, val_df, test_df)

    # Train VAE
    logger.info("➡️ Training VAE shard by shard...")
    shard_size = 5000
    vae = build_vae(X_train.shape[1])
    for start in range(0, len(X_train), shard_size):
        end = min(start + shard_size, len(X_train))
        shard = X_train[start:end]
        vae.fit(shard, shard,
                epochs=cfg.VAE_CONFIG['epochs'],
                batch_size=cfg.VAE_CONFIG['batch_size'],
                validation_data=(X_val, X_val),
                verbose=0)
    vae.save(os.path.join(CHECKPOINT_DIR, "vae_model.keras"))
    logger.info("✅ VAE trained and saved.")

    # Train Isolation Forest
    logger.info("➡️ Training Isolation Forest...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_train)
    joblib.dump(iso, cfg.ISOLATION_FOREST_MODEL_PATH)

    # Train KMeans
    logger.info("➡️ Training KMeans...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    joblib.dump(kmeans, cfg.KMEANS_MODEL_PATH)

    # Evaluate models
    logger.info("📊 Evaluating models...")
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)
    y_pred_kmeans = (kmeans.predict(X_test) == 1).astype(int)

    def report_scores(y_true, y_pred, model_name):
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        logger.info(f"{model_name} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return [model_name, prec, rec, f1, auc]

    rows = []
    rows.append(report_scores(y_test, y_pred_iso, "Isolation Forest"))
    rows.append(report_scores(y_test, y_pred_kmeans, "KMeans"))

    pd.DataFrame(rows, columns=["Model", "Precision", "Recall", "F1-score", "AUC"]).to_csv(REPORT_FILE, index=False)
    logger.info(f"📁 Report saved in {REPORT_FILE}")

if __name__ == "__main__":
    train_models()