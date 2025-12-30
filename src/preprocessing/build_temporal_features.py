# src/preprocessing/build_temporal_features.py
"""
===============================================================================
Build Temporal Features from Event-Level Data using Sliding Windows
===============================================================================

This script prepares temporal (process-level) features from event-level
dynamic data using sliding windows for unsupervised anomaly detection models
(VAE, Isolation Forest, KMeans).

Pipeline steps:
1. Load event-level CSV datasets (training, validation, testing)
2. Apply sliding windows per process
3. Aggregate event-level features into temporal features:
   - event_count
   - unique_event_count
   - avg_stack_depth
   - max_stack_depth
   - error_ratio
   - cluster_switch_rate
4. Handle missing values by zero-imputation
5. Standardize features using StandardScaler (fit on training data only)
6. Persist processed datasets and scaler for reproducibility and inference

------------------------------------------------------------------------------
Input:
- Event-level CSV files containing process execution data:
    - timestamp
    - processId
    - threadId
    - parentProcessId
    - userId
    - eventId
    - argsNum
    - returnValue
    - behavior_cluster
    - stack_depth
    - other optional columns

------------------------------------------------------------------------------
Output:
- X_train.npy : standardized temporal training feature matrix
- X_val.npy   : standardized temporal validation feature matrix
- X_test.npy  : standardized temporal testing feature matrix
- scaler.pkl  : fitted StandardScaler object

All outputs are saved under:
    <PROCESSED_DYNAMIC_DIR>/temporal/

------------------------------------------------------------------------------
Notes:
- This script builds higher-level temporal features from event-level data.
- Inputs are the same CSVs used in `01_prepare_event_level_unlabeled.py`.
- Reproducibility is ensured by fitting the scaler only on the training data.

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

OUTPUT_DIR = os.path.join(config.PROCESSED_DYNAMIC_DIR, "temporal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 5.0  # seconds
STEP_SIZE = 1.0    # seconds

# Event-level features to aggregate
EVENT_FEATURES = [
    "eventId",
    "stack_depth",
    "behavior_cluster",
    "returnValue"
]

def aggregate_window(df_window):
    """Compute temporal features for one window of events"""
    features = {}
    features["event_count"] = len(df_window)
    features["unique_event_count"] = df_window["eventId"].nunique()
    features["avg_stack_depth"] = df_window["stack_depth"].mean()
    features["max_stack_depth"] = df_window["stack_depth"].max()
    features["error_ratio"] = (df_window["returnValue"] != 0).mean()
    
    # Cluster switch rate
    clusters = df_window["behavior_cluster"].to_numpy()
    if len(clusters) > 1:
        switches = np.sum(clusters[1:] != clusters[:-1])
        features["cluster_switch_rate"] = switches / (len(clusters)-1)
    else:
        features["cluster_switch_rate"] = 0.0
    
    return features

def build_temporal_features(csv_file):
    """Convert event-level CSV to temporal features using sliding windows"""
    df = pd.read_csv(csv_file)
    df = df.sort_values(["processId", "timestamp"])
    
    temporal_rows = []
    for pid, group in df.groupby("processId"):
        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()
        t = start_time
        while t <= end_time:
            window_df = group[(group["timestamp"] >= t) & (group["timestamp"] < t + WINDOW_SIZE)]
            if len(window_df) > 0:
                features = aggregate_window(window_df)
                features["processId"] = pid
                features["window_start"] = t
                temporal_rows.append(features)
            t += STEP_SIZE
    return pd.DataFrame(temporal_rows)

def prepare_temporal_data():
    print("📥 Building temporal features...")
    
    train_temporal = build_temporal_features(config.DYNAMIC_FILE)
    val_temporal = build_temporal_features(os.path.join(config.RAW_DATA_DIR, "labelled_validation_data.csv"))
    test_temporal = build_temporal_features(os.path.join(config.RAW_DATA_DIR, "labelled_testing_data.csv"))
    
    # Drop identifiers, only features for model input
    feature_cols = ["event_count", "unique_event_count", "avg_stack_depth",
                    "max_stack_depth", "error_ratio", "cluster_switch_rate"]
    
    X_train = train_temporal[feature_cols].fillna(0).to_numpy()
    X_val = val_temporal[feature_cols].fillna(0).to_numpy()
    X_test = test_temporal[feature_cols].fillna(0).to_numpy()
    
    # Standardization
    print("⚖️ Standardizing temporal features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    
    print(f"💾 Temporal features saved in {OUTPUT_DIR}")
    print(f"   - X_train: {X_train_scaled.shape}")
    print(f"   - X_val: {X_val_scaled.shape}")
    print(f"   - X_test: {X_test_scaled.shape}")

if __name__ == "__main__":
    prepare_temporal_data()

# # src/preprocessing/02_build_temporal_features.py
# """
# ===============================================================================
# Build Temporal Behavioral Features from Event-Level Data (Sliding Windows)
# ===============================================================================

# This script aggregates event-level system activity logs into process-centric
# temporal behavioral features using sliding windows. The resulting matrices
# capture execution dynamics and are used for unsupervised anomaly detection.

# Aggregation strategy:
# - Window size: 5 seconds
# - Step size: 1 second
# - Features computed per window per process:
#     event_count, unique_event_count,
#     avg_stack_depth, max_stack_depth,
#     error_ratio, cluster_switch_rate
# - Standardization using StandardScaler (fit on training set)

# Input:
# - Structured CSV files with event-level features and timestamps

# Output:
# - Process-level temporal feature matrices suitable for anomaly detection
# - Saved as X_train.npy, X_val.npy, X_test.npy + scaler.pkl

# Author: Ngueyep Ulrich
# Date: 2025-12-29
# ===============================================================================
# """

# import os
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler
# from config.config import Config
# from tqdm import tqdm

# # ------------------------------
# # Initialization
# # ------------------------------
# config = Config()
# config.ensure_dirs()

# OUTPUT_DIR = os.path.join(config.PROCESSED_DYNAMIC_DIR, "temporal")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# WINDOW_SIZE = 5.0  # seconds
# STEP_SIZE = 1.0    # seconds

# TEMPORAL_FEATURES = [
#     "event_count",
#     "unique_event_count",
#     "avg_stack_depth",
#     "max_stack_depth",
#     "error_ratio",
#     "cluster_switch_rate"
# ]

# # ------------------------------
# # Helper functions
# # ------------------------------
# def compute_features(df_window):
#     if df_window.empty:
#         return [0] * len(TEMPORAL_FEATURES)
    
#     event_count = len(df_window)
#     unique_event_count = df_window['eventId'].nunique()
#     avg_stack_depth = df_window['stack_depth'].mean()
#     max_stack_depth = df_window['stack_depth'].max()
#     error_ratio = (df_window['returnValue'] != 0).mean()
    
#     # cluster_switch_rate: proportion of consecutive events where behavior_cluster changes
#     clusters = df_window['behavior_cluster'].values
#     if len(clusters) <= 1:
#         cluster_switch_rate = 0.0
#     else:
#         cluster_switch_rate = np.mean(clusters[1:] != clusters[:-1])
    
#     return [
#         event_count,
#         unique_event_count,
#         avg_stack_depth,
#         max_stack_depth,
#         error_ratio,
#         cluster_switch_rate
#     ]

# def sliding_window_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
#     process_ids = df['processId'].unique()
#     features_list = []
    
#     for pid in tqdm(process_ids, desc="Processing processes"):
#         df_proc = df[df['processId'] == pid].sort_values('timestamp')
#         start_time = df_proc['timestamp'].min()
#         end_time = df_proc['timestamp'].max()
        
#         t = start_time
#         while t + window_size <= end_time:
#             window_df = df_proc[(df_proc['timestamp'] >= t) & (df_proc['timestamp'] < t + window_size)]
#             feats = compute_features(window_df)
#             features_list.append(feats)
#             t += step_size
    
#     return np.array(features_list)

# # ------------------------------
# # Main function
# # ------------------------------
# def build_temporal_features():
#     print("📥 Loading CSV files...")
    
#     train_df = pd.read_csv(config.DYNAMIC_FILE)
#     val_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "labelled_validation_data.csv"))
#     test_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "labelled_testing_data.csv"))
    
#     print(f"✅ Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
#     # Compute features
#     print("⚡ Computing temporal features with sliding windows...")
#     X_train = sliding_window_features(train_df)
#     X_val = sliding_window_features(val_df)
#     X_test = sliding_window_features(test_df)
    
#     # Standardization
#     print("⚖️ Standardizing features using StandardScaler...")
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Save results
#     np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
#     np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val_scaled)
#     np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
#     joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    
#     print(f"💾 Temporal features saved in {OUTPUT_DIR}")
#     print(f"   - X_train: {X_train_scaled.shape}")
#     print(f"   - X_val: {X_val_scaled.shape}")
#     print(f"   - X_test: {X_test_scaled.shape}")

# if __name__ == "__main__":
#     build_temporal_features()
