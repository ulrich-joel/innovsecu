# src/preprocessing/prepare_event_level_unlabeled.py
"""
===============================================================================
 Prepare Event-Level Dynamic Data for Unsupervised Anomaly Detection
===============================================================================

This script prepares event-level dynamic data for unsupervised anomaly
detection models (Variational Autoencoder, Isolation Forest, KMeans).

Pipeline steps:
1. Load CSV datasets (training, validation, testing)
2. Select event-level behavioral features available in the dataset
3. Handle missing values by zero-imputation
4. Standardize features using StandardScaler (fit on training data only)
5. Persist processed datasets and scaler for reproducibility and inference

------------------------------------------------------------------------------
Input:
- Raw CSV files containing event-level system activity data
  (e.g., process execution context, system call metadata)

Selected features:
- processId
- threadId
- parentProcessId
- userId
- eventId
- argsNum
- returnValue
- behavior_cluster
- stack_depth

------------------------------------------------------------------------------
Output:
- X_train.npy : standardized training feature matrix
- X_val.npy   : standardized validation feature matrix
- X_test.npy  : standardized testing feature matrix
- scaler.pkl  : fitted StandardScaler object

All outputs are saved under:
    <PROCESSED_DYNAMIC_DIR>/unlabeled/

------------------------------------------------------------------------------
Important Note:
The current dataset provides event-level features only.
Higher-level temporal and behavioral features (e.g., rates, entropies,
temporal aggregates) are derived in a separate preprocessing stage.
This design ensures a clear separation between raw event preparation
and advanced feature engineering.

------------------------------------------------------------------------------
Reproducibility:
- Scaling parameters are learned exclusively from training data
- Validation and test sets are transformed using the same scaler

Author: Ngueyep Ulrich
Date: 2025-12-29
===============================================================================
"""


import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from config.config import Config

# ------------------------------
# Initialization
# ------------------------------
config = Config()
config.ensure_dirs()

OUTPUT_DIR = os.path.join(config.PROCESSED_DYNAMIC_DIR, "unlabeled")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Define dynamic features
# ------------------------------

STATIC_FEATURES = [
    "processId", "threadId", "parentProcessId",
    "userId", "eventId", "argsNum",
    "returnValue", "behavior_cluster", "stack_depth"
]

# FILE_FEATURES = [
#     "file_create_rate",
#     "file_delete_rate",
#     "file_write_bytes",
#     "file_read_bytes",
#     "file_name_entropy",
#     "file_extension_entropy",
#     "avg_file_size",
#     "file_handle_count"
# ]

# PROCESS_FEATURES = [
#     "cpu_usage_mean",
#     "cpu_usage_std",
#     "memory_usage_mean",
#     "memory_usage_std",
#     "process_spawn_rate",
#     "thread_count_mean"
# ]

# NETWORK_FEATURES = [
#     "unique_ip_count",
#     "outbound_conn_rate",
#     "inbound_conn_rate",
#     "bytes_sent",
#     "bytes_received",
#     "connection_duration_mean"
# ]

# SYSTEM_FEATURES = [
#     "registry_write_rate",
#     "new_service_created",
#     "privilege_escalation_attempts",
#     "abnormal_file_handle_count",
#     "failed_process_launch"
# ]

# TEMPORAL_FEATURES = [
#     "delta_file_create_rate",
#     "delta_cpu_usage",
#     "zscore_file_write_bytes",
#     "entropy_ratio_file_names",
#     "rolling_avg_process_spawn"
# ]

DYNAMIC_FEATURES = (
    STATIC_FEATURES
    # + FILE_FEATURES
    # + PROCESS_FEATURES
    # + NETWORK_FEATURES
    # + SYSTEM_FEATURES
    # + TEMPORAL_FEATURES
)

# ------------------------------
# Main function
# ------------------------------
def prepare_dynamic_data():
    print("📥 Loading CSV files...")
    
    train_df = pd.read_csv(config.DYNAMIC_FILE)
    val_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "labelled_validation_data.csv"))
    test_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "labelled_testing_data.csv"))
    
    print(f"✅ Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Check all features exist
    for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        missing = [f for f in DYNAMIC_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"❌ {df_name} CSV missing these columns: {missing}")
    
    # Select features
    X_train = train_df[DYNAMIC_FEATURES].fillna(0).to_numpy()
    X_val = val_df[DYNAMIC_FEATURES].fillna(0).to_numpy()
    X_test = test_df[DYNAMIC_FEATURES].fillna(0).to_numpy()
    
    # Standardization
    print("⚖️ Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save files
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    
    print(f"💾 Data prepared and saved in {OUTPUT_DIR}")
    print(f"   - X_train: {X_train_scaled.shape}")
    print(f"   - X_val: {X_val_scaled.shape}")
    print(f"   - X_test: {X_test_scaled.shape}")

if __name__ == "__main__":
    prepare_dynamic_data()
