# src/detection/meta/build_window_labels.py
"""
===============================================================================
Build Window-Level Labels aligned with Temporal Features
===============================================================================

This script generates window-level labels that are STRICTLY aligned with
the temporal windows used to build feature matrices.

A window is labeled as malicious (1) if at least one event inside the window
has evil == 1.

IMPORTANT:
- Window generation logic MUST match 02_build_temporal_features.py
- Same grouping (processId)
- Same window size and step
- Same empty-window filtering

===============================================================================
"""

import os
import numpy as np
import pandas as pd
from config.config import Config

# ------------------------------
# MUST match feature extraction
# ------------------------------
WINDOW_SIZE = 5.0
STEP_SIZE = 1.0

config = Config()

RAW_DIR = config.RAW_DATA_DIR
OUTPUT_DIR = config.RAW_DATA_DIR

# ------------------------------
# Window-aligned label builder
# ------------------------------
def build_labels(csv_path):
    df = pd.read_csv(csv_path)

    labels = []

    for pid, proc_df in df.groupby("processId"):
        proc_df = proc_df.sort_values("timestamp")

        t_min = proc_df["timestamp"].min()
        t_max = proc_df["timestamp"].max()

        t = t_min
        while t + WINDOW_SIZE <= t_max:
            window_df = proc_df[
                (proc_df["timestamp"] >= t) &
                (proc_df["timestamp"] < t + WINDOW_SIZE)
            ]

            # 🔥 MUST match feature logic
            if not window_df.empty:
                label = int((window_df["evil"] == 1).any())
                labels.append(label)

            t += STEP_SIZE

    return np.array(labels)

# ------------------------------
# Main
# ------------------------------
print("🏷️ Building ALIGNED window-level labels...")

y_train = build_labels(
    os.path.join(RAW_DIR, "labelled_training_data.csv")
)
y_val = build_labels(
    os.path.join(RAW_DIR, "labelled_validation_data.csv")
)
y_test = build_labels(
    os.path.join(RAW_DIR, "labelled_testing_data.csv")
)

print(f"✅ y_train: {y_train.shape}")
print(f"✅ y_val  : {y_val.shape}")
print(f"✅ y_test : {y_test.shape}")

np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("💾 Aligned window-level labels saved")
