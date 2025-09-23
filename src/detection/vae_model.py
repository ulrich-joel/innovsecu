import logging, os
import numpy as np
import torch, joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config.config import Config
from src.detection.detection_model import ModelTrainer
from src.evaluation.evaluate_dynamic_models import evaluate_models, save_models
from src.evaluation.dynamic_detection_models import fusion_evaluation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_trainer")

config = Config()
config.ensure_dirs()  # Ensure necessary directories exist

def load_and_normalize_data(cfg):
    logger.info("Chargement des fichiers numpy traités...")

    X_train = np.load(cfg.IF_KMEANS_TRAIN_FILE)
    X_val = np.load(cfg.IF_KMEANS_VAL_FILE)
    X_test = np.load(cfg.IF_KMEANS_TEST_FILE)
    y_test = np.load(os.path.join(cfg.PROCESSED_DYNAMIC_DIR, 'y_test.npy'))

    y_train = np.zeros(len(X_train))
    y_val = np.zeros(len(X_val))

    return X_train, X_val, X_test, y_train, y_val, y_test

def balance_test_set(X, y, method="downsample"):
    from sklearn.utils import resample

    X_0 = X[y == 0]
    y_0 = y[y == 0]
    X_1 = X[y == 1]
    y_1 = y[y == 1]

    if method == "downsample":
        X_1_down, y_1_down = resample(X_1, y_1, replace=False, n_samples=len(y_0), random_state=42)
        X_bal = np.vstack((X_0, X_1_down))
        y_bal = np.hstack((y_0, y_1_down))

    elif method == "upsample":
        X_0_up, y_0_up = resample(X_0, y_0, replace=True, n_samples=len(y_1), random_state=42)
        X_bal = np.vstack((X_0_up, X_1))
        y_bal = np.hstack((y_0_up, y_1))

    else:
        raise ValueError("Méthode inconnue")

    return X_bal, y_bal

def train_models():
    cfg = Config()
    device = "cpu"
    trainer = ModelTrainer(device=device)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_normalize_data(cfg)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=64)

    input_dim = train_tensor.shape[1]

    logger.info("➡️ Entraînement du VAE...")
    trainer.train_vae(train_loader, val_loader, input_dim=input_dim)

    logger.info("➡️ Entraînement de l'Isolation Forest...")
    trainer.train_isolation_forest(X_train)

    logger.info("➡️ Entraînement de KMeans...")
    trainer.train_kmeans(X_train)

    print("✔️ Dimensions au moment de l'évaluation :")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("📁 Fichier y_test chargé depuis :", os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_test.npy"))

    logger.info("🧪 Évaluation des modèles...")
    evaluate_models(trainer, X_test, y_test)

    # ✅ Rééquilibrage automatique avant la fusion
    logger.info("⚖️ Rééquilibrage du jeu de test avant fusion (downsampling)...")
    X_test_bal, y_test_bal = balance_test_set(X_test, y_test, method="downsample")
    print(f"🧪 Nouveau jeu équilibré : {np.bincount(y_test_bal)}")

    logger.info("🔁 Fusion des prédictions...")
    fusion_evaluation(trainer, X_test, y_test, balanced=False)   # Évaluation complète
    fusion_evaluation(trainer, X_test_bal, y_test_bal, balanced=True)    # Évaluation équilibrée

    logger.info("📊 Comparaison fusion full vs équilibrée...")
    df_full = pd.read_csv(os.path.join(config.REPORTS_DIR, "fusion_metrics_full.csv"))
    df_bal = pd.read_csv(os.path.join(config.REPORTS_DIR, "fusion_metrics_balanced.csv"))

    df_full["Type"] = "Full"
    df_bal["Type"] = "Balanced"

    df_comparatif = pd.concat([df_full, df_bal], ignore_index=True)
    comparison_path = os.path.join(config.REPORTS_DIR, "fusion_comparison.csv")
    df_comparatif.to_csv(comparison_path, index=False)
    print(f"📁 Comparaison fusion sauvegardée dans : {comparison_path}")

    logger.info("💾 Sauvegarde des modèles...")
    save_models(trainer, config)

if __name__ == "__main__":
    train_models()


# import logging, os
# import numpy as np
# import torch, joblib
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from config.config import Config
# from src.detection.detection_model import ModelTrainer
# from src.evaluation.evaluate_dynamic_models import evaluate_models, save_models
# from src.evaluation.dynamic_detection_models import fusion_evaluation


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("dynamic_trainer")

# config = Config()
# config.ensure_dirs()  # Ensure necessary directories exist

# def load_and_normalize_data(cfg):
#     logger.info("Chargement des fichiers numpy traités...")

#     X_train = np.load(cfg.IF_KMEANS_TRAIN_FILE)
#     X_val = np.load(cfg.IF_KMEANS_VAL_FILE)
#     X_test = np.load(cfg.IF_KMEANS_TEST_FILE)
#     y_test = np.load(os.path.join(cfg.PROCESSED_DYNAMIC_DIR, 'y_test.npy'))



#     # Labels d'entraînement et validation mis à 0 car non utilisés dans VAE/IF/KMeans
#     y_train = np.zeros(len(X_train))
#     y_val = np.zeros(len(X_val))

#     return X_train, X_val, X_test, y_train, y_val, y_test

# def train_models():
#     cfg = Config()
#     device = "cpu"
#     trainer = ModelTrainer(device=device)

#     X_train, X_val, X_test, y_train, y_val, y_test = load_and_normalize_data(cfg)

#     # Convert to tensors
#     train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     val_tensor = torch.tensor(X_val, dtype=torch.float32)

#     # Créer des DataLoaders pour VAE
#     train_dataset = TensorDataset(train_tensor)
#     val_dataset = TensorDataset(val_tensor)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64)

#     input_dim = train_tensor.shape[1]

#     logger.info("➡️ Entraînement du VAE...")
#     trainer.train_vae(train_loader, val_loader, input_dim=input_dim)

#     logger.info("➡️ Entraînement de l'Isolation Forest...")
#     trainer.train_isolation_forest(X_train)

#     logger.info("➡️ Entraînement de KMeans...")
#     trainer.train_kmeans(X_train)

#     print("✔️ Dimensions au moment de l'évaluation :")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"y_test shape: {y_test.shape}")
#     print("📁 Fichier y_test chargé depuis :", os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_test.npy"))



#     # logger.info("🧪 Évaluation des modèles...")
#     # evaluate_models(trainer, X_test, y_test)

#     # logger.info("💾 Sauvegarde des modèles...")
#     # save_models(trainer, config)

#     logger.info("🧪 Évaluation des modèles...")
#     evaluate_models(trainer, X_test, y_test)

#     logger.info("🔁 Fusion des prédictions...")
#     fusion_evaluation(trainer, X_test, y_test)

#     logger.info("💾 Sauvegarde des modèles...")
#     save_models(trainer, config)


# if __name__ == "__main__":
#     train_models()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import logging
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.cluster import KMeans
# import joblib
# from config.config import Config
# from sklearn.metrics import silhouette_score
# import os


# config = Config()

# class VAEModel(nn.Module):
#     def __init__(self, input_dim=15, latent_dim=8):
#         super(VAEModel, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(32, latent_dim)
#         self.fc_var = nn.Linear(32, latent_dim)
#         self.decoder_input = nn.Linear(latent_dim, 32)
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim)
#         )

#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_var(h)

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(self.decoder_input(z))

#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decode(z), mu, log_var

# class ModelTrainer:
#     def __init__(self, device="cpu", config=None):
#         self.logger = logging.getLogger(__name__)
#         self.device = device
#         self.model = None
#         self.isolation_forest = None
#         self.kmeans = None
#         self.config = config or Config()  # Make sure this line exists

#     def train_vae(self, train_tensor, val_tensor, config):
#         input_dim = config.VAE_CONFIG["input_dim"]
#         latent_dim = config.VAE_CONFIG["latent_dim"]
#         epochs = config.VAE_CONFIG["epochs"]
#         learning_rate = config.VAE_CONFIG["learning_rate"]
#         beta_max = config.VAE_CONFIG["beta_max"]
#         patience = config.VAE_CONFIG["patience"]

#         self.model = VAEModel(input_dim, latent_dim).to(self.device)
#         optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         criterion = nn.MSELoss()

#         best_val_loss = float('inf')
#         patience_counter = 0

#         for epoch in range(epochs):
#             self.model.train()
#             recon_x, mu, log_var = self.model(train_tensor)
#             recon_loss = criterion(recon_x, train_tensor)
#             kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(train_tensor)
#             beta = min(epoch / 10, 1.0) * beta_max
#             loss = recon_loss + beta * kl_div

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             self.model.eval()
#             with torch.no_grad():
#                 val_recon, val_mu, val_log_var = self.model(val_tensor)
#                 val_loss = criterion(val_recon, val_tensor) + beta * (
#                     -0.5 * torch.sum(1 + val_log_var - val_mu.pow(2) - val_log_var.exp()) / len(val_tensor))

#             self.logger.info(f"Epoch {epoch} - Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Beta: {beta:.4f}")

#             if val_loss.item() < best_val_loss:
#                 best_val_loss = val_loss.item()
#                 torch.save(self.model.state_dict(), config.BEST_VAE_MODEL_PATH)
#                 self.logger.info("Best VAE model saved")
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     self.logger.info("Early stopping triggered")
#                     break

#     def train_isolation_forest(self, X):
#         self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
#         self.isolation_forest.fit(X)
#         joblib.dump(self.isolation_forest, config.ISOLATION_FOREST_MODEL_PATH)
#         self.logger.info("Isolation Forest trained and saved")

#     def train_kmeans(self, X):
#         from sklearn.cluster import KMeans
#         from sklearn.metrics import silhouette_score

#         min_k = 2
#         max_k = 15
#         best_score = -1
#         best_k = None
#         best_model = None

#         kmeans_params = self.config.CLUSTERING_PARAMS

#         for k in range(min_k, max_k + 1):
#             # Override n_clusters dynamically
#             params = dict(kmeans_params)
#             params['n_clusters'] = k

#             model = KMeans(random_state=42, **params)
#             labels = model.fit_predict(X)
#             score = silhouette_score(X, labels)
#             self.logger.info(f"K={k}, Silhouette Score={score:.4f}")

#             if score > best_score:
#                 best_score = score
#                 best_k = k
#                 best_model = model

#         self.logger.info(f"Best K found: {best_k} with silhouette score: {best_score:.4f}")
#         self.kmeans = best_model

#         # Save the best model
#         import joblib
#         joblib.dump(best_model, self.config.KMEANS_MODEL_PATH)
#         self.logger.info(f"KMeans model saved to {self.config.KMEANS_MODEL_PATH}")