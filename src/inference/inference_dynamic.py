import torch
import numpy as np
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.config import Config
from src.detection.detection_model import VAEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

def load_new_data(path, features):
    df = pd.read_csv(path)
    X = df[features].values
    return X

def normalize_data(X, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(X)

def run_inference(X_input, config, device="cpu"):
    logger.info("🔁 Chargement des modèles sauvegardés...")

    # Load VAE
    vae = VAEModel(input_dim=X_input.shape[1], latent_dim=8).to(device)
    vae.load_state_dict(torch.load(config.BEST_VAE_MODEL_PATH, map_location=device))
    vae.eval()

    # Load Isolation Forest & KMeans
    iso_forest = joblib.load(config.BEST_ISOFOREST_MODEL_PATH)
    kmeans = joblib.load(config.BEST_KMEANS_MODEL_PATH)

    # ---- VAE Inference ----
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        recon_x, _, _ = vae(X_tensor)
        recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()
    vae_thresh = np.percentile(recon_errors, 95)
    vae_preds = (recon_errors > vae_thresh).astype(int)
    logger.info("✅ VAE - Seuil de reconstruction : %.4f", vae_thresh)

    # ---- Isolation Forest ----
    iso_preds = (iso_forest.predict(X_input) == -1).astype(int)

    # ---- KMeans ----
    distances = kmeans.transform(X_input).min(axis=1)
    km_thresh = np.percentile(distances, 95)
    km_preds = (distances > km_thresh).astype(int)
    logger.info("✅ KMeans - Seuil de distance : %.4f", km_thresh)

    return vae_preds, iso_preds, km_preds

if __name__ == "__main__":
    config = Config()
    device = "cpu"

    # ✅ Définis les mêmes features que lors de l'entraînement
    FEATURES = ['processId', 'threadId', 'parentProcessId', 'userId', 'eventId',
                'argsNum', 'returnValue', 'behavior_cluster', 'stack_depth']

    # ⚠️ Exemple : à adapter à ton fichier de données non étiquetées
    NEW_DATA_PATH = config.RAW_DATA_DIR + "/unseen_test_data.csv"
    SCALER_PATH = config.MODEL_DIR + "/scaler.pkl"  # À créer pendant l'entraînement

    logger.info("📥 Chargement des nouvelles données...")
    X_new = load_new_data(NEW_DATA_PATH, FEATURES)

    logger.info("🔄 Normalisation des données...")
    X_norm = normalize_data(X_new, SCALER_PATH)

    logger.info("🚨 Détection en cours...")
    vae_preds, iso_preds, km_preds = run_inference(X_norm, config, device)

    # 🔍 Affichage ou post-traitement
    df_result = pd.DataFrame({
        "VAE": vae_preds,
        "IsolationForest": iso_preds,
        "KMeans": km_preds
    })
    print(df_result.head())
