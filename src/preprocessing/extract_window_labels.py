# src/preprocessing/extract_window_labels.py
"""
===============================================================================
Extract Window-Level Labels with Stratified Split and Indices
===============================================================================

This script extracts binary labels (normal / malicious) at the window level,
performs a STRATIFIED train / validation / test split, and saves both:

- labels: y_train.npy, y_val.npy, y_test.npy
- indices: idx_train.npy, idx_val.npy, idx_test.npy

This guarantees correct alignment with meta-features for the Random Forest
meta-classifier training.

------------------------------------------------------------------------------ 
Input:
- labelled_training_data.csv
- labelled_validation_data.csv
- labelled_testing_data.csv

Each CSV must contain a binary column indicating malicious behavior.

------------------------------------------------------------------------------ 
Output (saved in outputs/processed/processed_dynamic/temporal/):
- y_train.npy, y_val.npy, y_test.npy
- idx_train.npy, idx_val.npy, idx_test.npy

Author: Ngueyep Ulrich
Date: 2025-12-30
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import Config

# ------------------------------
# Initialization
# ------------------------------
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

config = Config()

INPUT_DIR = config.RAW_DATA_DIR
OUTPUT_DIR = os.path.join(
    config.OUTPUT_DIR,
    "processed",
    "processed_dynamic",
    "temporal"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Helper functions
# ------------------------------
def load_labels_from_csv(csv_path, label_column="evil"):
    """
    Load binary labels from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    label_column : str
        Name of the column containing labels (0 = normal, 1 = malicious)

    Returns
    -------
    np.ndarray
        Array of binary labels
    """
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}")
    return df[label_column].values.astype(int)

def print_label_stats(name, y):
    """Print class distribution for a given split."""
    unique, counts = np.unique(y, return_counts=True)
    stats = dict(zip(unique, counts))
    print(f"{name}: {stats}")

# ------------------------------
# Load all labels
# ------------------------------
print("🔍 Extracting window labels from CSV files...")

y_train_raw = load_labels_from_csv(os.path.join(INPUT_DIR, "labelled_training_data.csv"))
y_val_raw   = load_labels_from_csv(os.path.join(INPUT_DIR, "labelled_validation_data.csv"))
y_test_raw  = load_labels_from_csv(os.path.join(INPUT_DIR, "labelled_testing_data.csv"))

# Concatenate all labels for stratified split
y_all = np.concatenate([y_train_raw, y_val_raw, y_test_raw])
print(f"Total windows: {len(y_all)}")
print_label_stats("Global", y_all)

# ------------------------------
# Stratified split
# ------------------------------
print("🔀 Performing stratified train / val / test split...")

indices = np.arange(len(y_all))

# First split: train vs temp (val + test)
idx_train, idx_temp = train_test_split(
    indices,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=y_all,
    random_state=SEED,
    shuffle=True
)

y_temp = y_all[idx_temp]

# Second split: validation vs test
idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=y_temp,
    random_state=SEED,
    shuffle=True
)

# ------------------------------
# Build final label arrays
# ------------------------------
y_train = y_all[idx_train]
y_val   = y_all[idx_val]
y_test  = y_all[idx_test]

# ------------------------------
# Sanity check
# ------------------------------
print("\n📊 Label statistics after split:")
print_label_stats("Train", y_train)
print_label_stats("Validation", y_val)
print_label_stats("Test", y_test)

if len(np.unique(y_train)) < 2:
    raise RuntimeError(
        "Training set contains only one class. "
        "Random Forest meta-classifier cannot be trained."
    )

# ------------------------------
# Save labels and indices
# ------------------------------
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

np.save(os.path.join(OUTPUT_DIR, "idx_train.npy"), idx_train)
np.save(os.path.join(OUTPUT_DIR, "idx_val.npy"), idx_val)
np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"), idx_test)

print(f"\n💾 Labels and indices saved in {OUTPUT_DIR}")
print("✅ Stratified split completed successfully.")
