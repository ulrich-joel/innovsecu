import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.config import Config
from src.detection.model_trainer import ModelTrainer
from sklearn.ensemble import IsolationForest
import joblib  # Ajoute ceci si ce n’est pas déjà importé



def prepare_static_data(csv_path: str, config: Config):
    """Load and preprocess the static dataset."""
    print(f"📥 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop unnecessary columns
    df.drop(columns=["FileName", "md5Hash"], inplace=True, errors='ignore')

    # Handle categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Extract features and labels
    if "Benign" not in df.columns:
        raise ValueError("🛑 'Benign' column not found in the dataset.")
    
    y = df["Benign"].astype(int)
    X = df.drop(columns=["Benign"])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler mean (optional)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    np.save(os.path.join(config.MODELS_DIR, "static_scaler_mean.npy"), scaler.mean_)

    print(f"✅ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features.")
    return X_scaled, y


def split_and_save_data(X, y, config: Config):
    """Split data into train/val/test and save them."""
    print("🔀 Splitting data into train/val/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Ensure directory exists
    os.makedirs(config.PROCESSED_STATIC_DIR, exist_ok=True)

    # Save splits
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_val.npy"), X_val)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_test.npy"), y_test)

    print("💾 Data splits saved to:", config.PROCESSED_STATIC_DIR)
    return X_train, X_val, y_train, y_val


def main():
    config = Config()
    config.ensure_dirs()

    print("🚀 Starting static model training pipeline...")

    # Preprocessing
    X_scaled, y = prepare_static_data(config.RAW_STATIC_DATA, config)
    X_train, X_val, y_train, y_val = split_and_save_data(X_scaled, y, config)

    print("🧠 Initializing model trainer...")
    trainer = ModelTrainer(device="cpu", config=config)


    # Isolation Forest avec plusieurs niveaux de contamination
    contamination_values = [0.01, 0.05, 0.1, 0.15]
    for contamination in contamination_values:
        print(f"⚙️ Training Isolation Forest with contamination = {contamination}")
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_train)

        model_path = os.path.join(config.MODELS_DIR, f"isolation_forest_{int(contamination*100)}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

    # KMeans
    print("⚙️ Training KMeans...")
    trainer.train_kmeans(X_train)

    print("📊 Résumé des modèles Isolation Forest sauvegardés :")
    for contamination in contamination_values:
        print(f" - isolation_forest_{int(contamination*100)}.pkl")

    print("✅ All static models trained and saved successfully.")



if __name__ == "__main__":
    main()


# #src/preprocessing/train_static_models.py
# import torch
# import logging
# import numpy as np
# from config.config import Config
# from src.detection.vae_model import ModelTrainer

# def main():
#     logging.basicConfig(level=logging.INFO)
#     config = Config()
#     trainer = ModelTrainer(device="cpu")

#     # Load the full VAE data
#     full_tensor = torch.load(config.VAE_DATA_FILE)

#     # Dynamic split: train / validation
#     train_size = int(0.8 * len(full_tensor))
#     val_size = len(full_tensor) - train_size
#     train_tensor, val_tensor = torch.utils.data.random_split(full_tensor, [train_size, val_size])

#     # Extract actual tensors (optional depending on PyTorch version)
#     train_tensor = train_tensor.dataset[train_tensor.indices] if hasattr(train_tensor, 'indices') else train_tensor
#     val_tensor = val_tensor.dataset[val_tensor.indices] if hasattr(val_tensor, 'indices') else val_tensor

#     logging.info("✅ Training Variational Autoencoder (VAE)...")
#     trainer.train_vae(train_tensor, val_tensor, config)

#     logging.info("✅ Training Isolation Forest...")
#     x_train = np.load(config.IF_KMEANS_TRAIN_FILE)
#     trainer.train_isolation_forest(x_train)

#     logging.info("✅ Training KMeans...")
#     trainer.train_kmeans(x_train)

# if __name__ == "__main__":
#     main()
