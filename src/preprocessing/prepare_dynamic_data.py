# prepare_dynamic_datasets.py

"""

==========================================================================================================
 Dynamic Model Preprocesses Script
==========================================================================================================

This script preprocesses labeled dynamic (runtime) datasets for anomaly detection or classification tasks.

It performs the following operations:
1. Loads training, validation, and testing CSV files from the dynamic dataset directory.
2. Selects a predefined set of numerical features and the label column.
3. Applies standard normalization using StandardScaler.
4. Saves the processed datasets and corresponding labels as `.npy` files.
5. Saves the fitted scaler for consistent transformation during inference or evaluation.

==========================================================================================================
Outputs:
- Scaled feature arrays: X_train, X_val, X_test
- Label arrays: y_train, y_val, y_test
- Scaler object for normalization
==========================================================================================================

Author: Ngueyep Ulrich
Date: 2025-10-14
==========================================================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import Config
import joblib

def prepare_dynamic_datasets(config: Config):
    """
    Prepare and preprocess dynamic (runtime) datasets for anomaly detection.

    Steps:
    - Loads labeled training, validation, and testing CSV files.
    - Extracts numerical features and binary labels.
    - Applies standard normalization on features.
    - Saves the resulting arrays and the scaler for later use.

    Args:
        config (Config): Configuration object with paths to input and output directories.

    Raises:
        ValueError: If required features or label columns are missing from the datasets.
    """

    base_path = os.path.join(config.PROCESSED_DYNAMIC_DIR, "intermediate")

    # Define input file paths
    train_path = os.path.join(base_path, "labelled_training_data.csv")
    val_path = os.path.join(base_path, "labelled_validation_data.csv")
    test_path = os.path.join(base_path, "labelled_testing_data.csv")

    # Columns to retain
    selected_features = [
        "timestamp", "processId", "threadId", "parentProcessId", "userId",
        "mountNamespace", "eventId", "argsNum", "returnValue", "stack_depth", "behavior_cluster"
    ]
    label_column = "sus"  # Label column (can be switched to 'evil' if needed)

    # Load datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Ensure all required columns exist
    for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        for col in selected_features + [label_column]:
            if col not in df.columns:
                raise ValueError(f"❌ Missing column '{col}' in {df_name}_df")

    # Extract features and labels
    X_train = train_df[selected_features].values
    y_train = train_df[label_column].values

    X_val = val_df[selected_features].values
    y_val = val_df[label_column].values

    X_test = test_df[selected_features].values
    y_test = test_df[label_column].values

    # Sanity checks
    assert X_train.shape[0] == y_train.shape[0], f"❌ Train set size mismatch: X={X_train.shape[0]}, y={y_train.shape[0]}"
    assert X_val.shape[0] == y_val.shape[0], f"❌ Validation set size mismatch: X={X_val.shape[0]}, y={y_val.shape[0]}"
    assert X_test.shape[0] == y_test.shape[0], f"❌ Test set size mismatch: X={X_test.shape[0]}, y={y_test.shape[0]}"

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save processed features
    np.save(config.IF_KMEANS_TRAIN_FILE, X_train_scaled)
    np.save(config.IF_KMEANS_VAL_FILE, X_val_scaled)
    np.save(config.IF_KMEANS_TEST_FILE, X_test_scaled)

    # Save labels
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_test.npy"), y_test)

    # Save scaler for reuse
    joblib.dump(scaler, os.path.join(config.PROCESSED_DYNAMIC_DIR, "scaler.pkl"))

    print("Dynamic datasets prepared and saved successfully.")


if __name__ == "__main__":
    cfg = Config()
    prepare_dynamic_datasets(cfg)

    # Display class distribution in y_test
    y_test = np.load(os.path.join(cfg.PROCESSED_DYNAMIC_DIR, "y_test.npy"))
    unique, counts = np.unique(y_test, return_counts=True)

    print("✔️ y_test class distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples")
