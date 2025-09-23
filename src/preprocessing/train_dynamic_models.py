# src/preprocessing/train_dynamic_models.py

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

# === Configuration initiale ===
cfg = Config()
cfg.ensure_dirs()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_trainer")
CHECKPOINT_DIR = cfg.MODELS_DIR
REPORT_FILE = os.path.join(cfg.REPORTS_DIR, "dynamic_model_report.csv")
os.makedirs(cfg.REPORTS_DIR, exist_ok=True)

features = ['processId', 'threadId', 'parentProcessId', 'userId', 'eventId', 'argsNum', 'returnValue', 'behavior_cluster', 'stack_depth']
label_col = 'sus'

def load_data():
    logger.info("Chargement des fichiers...")
    train = pd.read_csv(cfg.DYNAMIC_FILE)
    val = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, "labelled_validation_data.csv"))
    test = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, "labelled_testing_data.csv"))
    return train, val, test

def normalize(train, val, test):
    logger.info("Normalisation des données...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    return X_train, X_val, X_test, train[label_col], val[label_col], test[label_col]

# === VAE avec β-scheduler ===
class BetaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_beta, total_epochs):
        self.max_beta = max_beta
        self.total_epochs = total_epochs
        self.beta = 0

    def on_epoch_end(self, epoch, logs=None):
        self.beta = min(self.max_beta, (epoch + 1) / self.total_epochs * self.max_beta)

def build_vae(input_dim, encoding_dim=4):
    # --- Encodeur ---
    inputs = Input(shape=(input_dim,))
    h = Dense(16, activation='relu')(inputs)
    z_mean = Dense(encoding_dim, name="z_mean")(h)
    z_log_var = Dense(encoding_dim, name="z_log_var")(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], encoding_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])

    # --- Décodeur ---
    decoder_h = Dense(16, activation='relu')(z)
    outputs = Dense(input_dim, activation='linear')(decoder_h)

    # --- Modèle complet ---
    vae = Model(inputs, outputs)

    # --- Perte personnalisée ---
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer='adam')

    return vae


# === Entraînement shard par shard ===
def train_models():
    train_df, val_df, test_df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = normalize(train_df, val_df, test_df)

    logger.info("➡️ Entraînement du VAE shard par shard...")
    shard_size = 5000
    vae, scheduler = build_vae(X_train.shape[1])
    for start in range(0, len(X_train), shard_size):
        end = min(start + shard_size, len(X_train))
        shard = X_train[start:end]
        vae.fit(shard, shard,
                epochs=cfg.VAE_CONFIG['epochs'],
                batch_size=cfg.VAE_CONFIG['batch_size'],
                validation_data=(X_val, X_val),
                callbacks=[scheduler],
                verbose=0)
    vae.save(os.path.join(CHECKPOINT_DIR, "vae_model.keras"))
    logger.info("✅ VAE entraîné et sauvegardé.")

    logger.info("➡️ Entraînement Isolation Forest...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_train)
    joblib.dump(iso, cfg.ISOLATION_FOREST_MODEL_PATH)

    logger.info("➡️ Entraînement KMeans...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    joblib.dump(kmeans, cfg.KMEANS_MODEL_PATH)

    # === Évaluation ===
    logger.info("📊 Évaluation des modèles...")
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
    logger.info(f"📁 Rapport enregistré dans {REPORT_FILE}")

if __name__ == "__main__":
    train_models()


# # src/scripts/train_dynamic_models.py
# import numpy as np
# from config.config import Config
# from src.detection.vae_model import ModelTrainer

# def main():
#     config = Config()
#     dynamic_file = config.DYNAMIC_IF_KMEANS_TRAIN_FILE
#     data = np.load(dynamic_file)

#     trainer = ModelTrainer()

#     print("✅ Training Isolation Forest on dynamic data...")
#     trainer.train_isolation_forest(data)

#     print("✅ Training KMeans on dynamic data...")
#     trainer.train_kmeans(data)

# if __name__ == "__main__":
#     main()
