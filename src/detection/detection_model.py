import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from config.config import Config

config = Config()
config.ensure_dirs()

logger = logging.getLogger(__name__)

class VAEModel(nn.Module):
    def __init__(self, input_dim=15, latent_dim=32):
        super(VAEModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(64, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class ModelTrainer:
    def __init__(self, device="cpu"):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.isolation_forest = None
        self.kmeans = None
        self.config = config or Config()

    def train_vae(self, train_loader, val_loader, input_dim, latent_dim=8, epochs=50):
        self.logger.info(f"Initializing VAE with input_dim={input_dim}, latent_dim={latent_dim}")

        try:
            self.model = VAEModel(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)

            # Fix: No 'verbose' arg in ReduceLROnPlateau for older versions
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            best_val_loss = float('inf')
            patience_counter = 0
            patience_limit = 10

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                recon_loss_total = 0
                kl_loss_total = 0
                num_batches = 0

                for batch in train_loader:
                    x = batch[0].to(self.device)

                    optimizer.zero_grad()
                    recon_x, mu, log_var = self.model(x)

                    recon_loss = nn.MSELoss()(recon_x, x)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

                    beta = min(epoch / 10, 1.0) * 0.5
                    loss = recon_loss + beta * kl_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    recon_loss_total += recon_loss.item()
                    kl_loss_total += kl_loss.item()
                    num_batches += 1

                self.model.eval()
                val_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(self.device)
                        recon_x, mu, log_var = self.model(x)
                        recon_loss = nn.MSELoss()(recon_x, x)
                        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
                        val_loss += (recon_loss + beta * kl_loss).item()
                        val_batches += 1

                avg_train_loss = train_loss / num_batches
                avg_val_loss = val_loss / val_batches
                avg_recon_loss = recon_loss_total / num_batches
                avg_kl_loss = kl_loss_total / num_batches

                scheduler.step(avg_val_loss)

                self.logger.info(
                    f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} "
                    f"(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}) | "
                    f"Val Loss: {avg_val_loss:.4f} | β: {beta:.4f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), config.BEST_VAE_MODEL_PATH)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        self.logger.info("✅ Early stopping triggered.")
                        break

        except Exception as e:
            self.logger.error(f"Error during VAE training: {str(e)}")
            raise

    def train_isolation_forest(self, train_data):
        try:
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.isolation_forest.fit(train_data)
            self.logger.info("Isolation Forest successfully trained.")
        except Exception as e:
            self.logger.error(f"Error during Isolation Forest training: {str(e)}")
            raise

    def train_kmeans(self, train_data, n_clusters=2):
        try:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.kmeans.fit(train_data)
            self.logger.info("K-Means successfully trained.")
        except Exception as e:
            self.logger.error(f"Error during K-Means training: {str(e)}")
            raise

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import logging
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.cluster import KMeans
# from config.config import Config

# config = Config()
# config.ensure_dirs()  # Ensure necessary directories exist

# class VAEModel(nn.Module):
#     def __init__(self, input_dim=15, latent_dim=32):
#         super(VAEModel, self).__init__()
        
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Sequential(
#                 nn.Linear(input_dim, 64),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Linear(64, 32),
#                 nn.BatchNorm1d(32),
#                 nn.ReLU()
#             )
#         )
        
#         # Latent space
#         self.fc_mu = nn.Linear(32, latent_dim)
#         self.fc_var = nn.Linear(32, latent_dim)
        
#         # Decoder
#         self.decoder_input = nn.Linear(latent_dim, 32)
        
#         self.decoder = nn.Sequential(
#             nn.Sequential(
#                 nn.Linear(32, 64),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU()
#             )
#         )
        
#         self.final_layer = nn.Linear(64, input_dim)
        
#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_var(h)
        
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
        
#     def decode(self, z):
#         h = self.decoder_input(z)
#         h = self.decoder(h)
#         return self.final_layer(h)
        
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decode(z), mu, log_var

# class ModelTrainer:
#     def __init__(self, device="cpu"):
#         self.logger = logging.getLogger(__name__)
#         self.device = device
#         self.model = None
#         self.isolation_forest = None
#         self.kmeans = None
#         self.config = config or Config()

#     def train_vae(self, train_loader, val_loader, input_dim, latent_dim=8, epochs=50):
#         """Train the VAE with improved parameters"""
#         self.logger.info(f"Initializing VAE with input_dim={input_dim}, latent_dim={latent_dim}")
        
#         try:
#             # Model initialization
#             self.model = VAEModel(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
            
#             # Optimizer with learning rate scheduler
#             optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
#             scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', factor=0.5, patience=5
#             )
            
#             best_val_loss = float('inf')
#             patience_counter = 0
#             patience_limit = 10
            
#             for epoch in range(epochs):
#                 # Training
#                 self.model.train()
#                 train_loss = 0
#                 recon_loss_total = 0
#                 kl_loss_total = 0
#                 num_batches = 0
                
#                 for batch in train_loader:
#                     x = batch['features'].to(self.device)
                    
#                     optimizer.zero_grad()
#                     recon_x, mu, log_var = self.model(x)
                    
#                     # Separate losses
#                     recon_loss = nn.MSELoss()(recon_x, x)
#                     kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    
#                     # Beta-VAE with adaptive beta
#                     beta = min(epoch / 10, 1.0) * 0.5
#                     loss = recon_loss + beta * kl_loss
                    
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                     optimizer.step()
                    
#                     train_loss += loss.item()
#                     recon_loss_total += recon_loss.item()
#                     kl_loss_total += kl_loss.item()
#                     num_batches += 1
                
#                 # Validation
#                 self.model.eval()
#                 val_loss = 0
#                 val_batches = 0
                
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         x = batch['features'].to(self.device)
#                         recon_x, mu, log_var = self.model(x)
#                         recon_loss = nn.MSELoss()(recon_x, x)
#                         kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#                         val_loss += (recon_loss + beta * kl_loss).item()
#                         val_batches += 1
                
#                 # Compute averages
#                 avg_train_loss = train_loss / num_batches
#                 avg_val_loss = val_loss / val_batches
#                 avg_recon_loss = recon_loss_total / num_batches
#                 avg_kl_loss = kl_loss_total / num_batches
                
#                 # Update scheduler
#                 scheduler.step(avg_val_loss)
                
#                 self.logger.info(
#                     f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f} '
#                     f'(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}), '
#                     f'Val Loss: {avg_val_loss:.4f}, Beta: {beta:.4f}'
#                 )
                
#                 # Early stopping
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     torch.save(self.model.state_dict(), config.BEST_VAE_MODEL_PATH)
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= patience_limit:
#                         self.logger.info("Early stopping triggered")
#                         break
                    
#         except Exception as e:
#             self.logger.error(f"Error during VAE training: {str(e)}")
#             raise
    
#     def train_isolation_forest(self, train_data):
#         """Train Isolation Forest"""
#         try:
#             self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
#             self.isolation_forest.fit(train_data)
#             self.logger.info("Isolation Forest successfully trained")
#         except Exception as e:
#             self.logger.error(f"Error during Isolation Forest training: {str(e)}")
#             raise
        
#     def train_kmeans(self, train_data, n_clusters=2):
#         """Train K-Means"""
#         try:
#             self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             self.kmeans.fit(train_data)
#             self.logger.info("K-Means successfully trained")
#         except Exception as e:
#             self.logger.error(f"Error during K-Means training: {str(e)}")
#             raise

# def train_all_models(train_loader, val_loader, input_dim):
#     """Train all detection models"""
#     logger = logging.getLogger(__name__)
#     trainer = ModelTrainer()
    
#     try:
#         # Prepare data for sklearn models
#         train_data = []
#         for batch in train_loader:
#             train_data.append(batch['features'].numpy())
#         train_data = np.concatenate(train_data)
        
#         # Train models
#         logger.info("Training VAE...")
#         trainer.train_vae(train_loader, val_loader, input_dim)
        
#         logger.info("Training Isolation Forest...")
#         trainer.train_isolation_forest(train_data)
        
#         logger.info("Training K-Means...")
#         trainer.train_kmeans(train_data)
        
#         return trainer
        
#     except Exception as e:
#         logger.error(f"Error during model training: {str(e)}")
#         raise
