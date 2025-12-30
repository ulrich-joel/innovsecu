# src/detection/unsupervised/train_vae.py
"""
===============================================================================
Train Variational Autoencoder (VAE) on Temporal Behavioral Features
===============================================================================

This script trains a Variational Autoencoder on temporal behavioral features.
The VAE learns a compact latent representation of normal behavior and uses
reconstruction error as an anomaly score.

------------------------------------------------------------------------------
Input:
- X_train.npy
- X_val.npy
- X_test.npy

------------------------------------------------------------------------------
Output:
- vae_model.pt
- reconstruction error scores for train / val / test

Author: Ngueyep Ulrich
Date: 2025-12-29
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config.config import Config

# ------------------------------
# Configuration
# ------------------------------
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
LATENT_DIM = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

INPUT_DIR = os.path.join(
    config.PROCESSED_DYNAMIC_DIR,
    "temporal"
)

OUTPUT_DIR = os.path.join(
    config.MODELS_DIR,
    "vae"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
print("📥 Loading temporal features...")

X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(INPUT_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(INPUT_DIR, "X_test.npy"))

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ------------------------------
# VAE Model
# ------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ------------------------------
# Loss function
# ------------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------------------
# Training
# ------------------------------
print("🧠 Training VAE...")

vae = VAE(input_dim=X_train.shape[1], latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LR)

vae.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = vae(x_batch)
        loss = vae_loss(recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

# ------------------------------
# Reconstruction error (scores)
# ------------------------------
def compute_reconstruction_error(model, X):
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        recon, _, _ = model(X)
        error = torch.mean((X - recon) ** 2, dim=1)
    return error.cpu().numpy()

scores_train = compute_reconstruction_error(vae, X_train)
scores_val   = compute_reconstruction_error(vae, X_val)
scores_test  = compute_reconstruction_error(vae, X_test)

# ------------------------------
# Save outputs
# ------------------------------
torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_model.pt"))

np.save(os.path.join(OUTPUT_DIR, "vae_scores_train.npy"), scores_train)
np.save(os.path.join(OUTPUT_DIR, "vae_scores_val.npy"), scores_val)
np.save(os.path.join(OUTPUT_DIR, "vae_scores_test.npy"), scores_test)

print("💾 VAE model and scores saved")
print(f"   - Scores train: {scores_train.shape}")
print(f"   - Scores val  : {scores_val.shape}")
print(f"   - Scores test : {scores_test.shape}")
