import numpy as np
import os
from config.config import Config

config = Config()
OUTPUT_DIR = os.path.join(config.PROCESSED_DYNAMIC_DIR, "temporal")

# Load temporal features
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))

# Load labels
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

# Check alignment
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# Quick sanity check: all shapes should match
assert X_train.shape[0] == y_train.shape[0], "Mismatch in training set!"
assert X_val.shape[0]   == y_val.shape[0],   "Mismatch in validation set!"
assert X_test.shape[0]  == y_test.shape[0],  "Mismatch in test set!"

print("✅ All temporal features and labels are perfectly aligned.")
