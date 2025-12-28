# src/preprocessing/prepare_static_models.py
"""
===============================================================================
 Static Dataset Preparation Script
===============================================================================

This module is responsible for:
1. Loading the raw static dataset from CSV.
2. Preprocessing the features (encoding, normalization).
3. Splitting the dataset into train, validation, and test sets.
4. Saving the processed datasets for model training.

===============================================================================
 Outputs:
 - Processed datasets: *.npy files (x_train, y_train, x_val, etc.)
 - Scaler mean (optional): static_scaler_mean.npy
===============================================================================

Author : Ngueyep Ulrich
Date   : 2025-10-14
===============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.config import Config

def prepare_static_data(csv_path: str, config: Config):
    """
    Load and preprocess the static dataset.

    Steps:
    - Drop unnecessary columns like 'FileName' and 'md5Hash' if present.
    - Encode categorical features using LabelEncoder.
    - Separate features (X) and binary labels (y).
    - Normalize features using StandardScaler.

    Args:
        csv_path (str): Path to the raw CSV file.
        config (Config): Project configuration.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled feature matrix (X), labels (y).
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop irrelevant columns
    df.drop(columns=["FileName", "md5Hash"], inplace=True, errors='ignore')

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Ensure the label column exists
    if "Benign" not in df.columns:
        raise ValueError("❌ Missing 'Benign' column in the dataset.")

    # Separate features and labels
    y = df["Benign"].astype(int)
    X = df.drop(columns=["Benign"])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optionally save the scaler mean
    os.makedirs(config.MODELS_DIR_STATIC_TRAIN, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.MODELS_DIR_STATIC_TRAIN, "meta_scaler.pkl"))
    np.save(os.path.join(config.MODELS_DIR_STATIC_TRAIN, "static_scaler_mean.npy"), scaler.mean_)

    print(f"Data preprocessed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features.")
    return X_scaled, y

def split_and_save_data(X, y, config: Config):
    """
    Split dataset into train, validation, and test sets and save them as .npy files.

    Splits:
    - 70% training
    - 15% validation
    - 15% testing

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        config (Config): Project configuration.

    Returns:
        Tuple of splits: X_train, X_val, y_train, y_val
    """
    print("Splitting dataset into train / validation / test...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    os.makedirs(config.PROCESSED_STATIC_DIR, exist_ok=True)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_val.npy"), X_val)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "x_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_STATIC_DIR, "y_test.npy"), y_test)

    print(f"✅ Data splits saved to: {config.PROCESSED_STATIC_DIR}")
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    config = Config()
    config.ensure_dirs()

    print("🚀 Lancement de la préparation du dataset statique...")

    X_scaled, y = prepare_static_data(config.RAW_STATIC_DATA, config)
    split_and_save_data(X_scaled, y, config)

    print("✅ Préparation et sauvegarde des données terminées.")
