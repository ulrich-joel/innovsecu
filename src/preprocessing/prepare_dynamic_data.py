import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import Config
import joblib

def prepare_dynamic_datasets(config: Config):
    """
    Prepare dynamic datasets for training, validation, and testing.

    This function:
    - Loads labeled training, validation, and testing data.
    - Selects specific numerical features and labels.
    - Normalizes the features using StandardScaler.
    - Saves the processed datasets and labels as .npy files.
    - Saves the scaler for future use.

    Args:
        config (Config): Configuration object containing paths and settings.

    Raises:
        ValueError: If any required column is missing in the datasets.
    """
    base_path = os.path.join(config.PROCESSED_DYNAMIC_DIR, "intermediate")

    # Source files
    train_path = os.path.join(base_path, "labelled_training_data.csv")
    val_path = os.path.join(base_path, "labelled_validation_data.csv")
    test_path = os.path.join(base_path, "labelled_testing_data.csv")

    # Columns to keep (numerical only)
    selected_features = [
        "timestamp", "processId", "threadId", "parentProcessId", "userId",
        "mountNamespace", "eventId", "argsNum", "returnValue", "stack_depth", "behavior_cluster"
    ]

    label_column = "sus"  # Can be replaced with 'evil' if needed

    # Load CSV files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Check for missing columns
    for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        for col in selected_features + [label_column]:
            if col not in df.columns:
                raise ValueError(f"❌ Column '{col}' is missing in {df_name}_df")

    # Extract features and labels
    X_train = train_df[selected_features].values
    y_train = train_df[label_column].values

    X_val = val_df[selected_features].values
    y_val = val_df[label_column].values

    X_test = test_df[selected_features].values
    y_test = test_df[label_column].values

    # 🔒 Check dimensions
    assert X_train.shape[0] == y_train.shape[0], f"❌ Train mismatch: X={X_train.shape[0]}, y={y_train.shape[0]}"
    assert X_val.shape[0] == y_val.shape[0], f"❌ Val mismatch: X={X_val.shape[0]}, y={y_val.shape[0]}"
    assert X_test.shape[0] == y_test.shape[0], f"❌ Test mismatch: X={X_test.shape[0]}, y={y_test.shape[0]}"

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save processed datasets
    np.save(config.IF_KMEANS_TRAIN_FILE, X_train_scaled)
    np.save(config.IF_KMEANS_VAL_FILE, X_val_scaled)
    np.save(config.IF_KMEANS_TEST_FILE, X_test_scaled)

    # Save labels
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.PROCESSED_DYNAMIC_DIR, "y_test.npy"), y_test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(config.PROCESSED_DYNAMIC_DIR, "scaler.pkl"))

    print("✅ Dynamic datasets prepared and saved.")

if __name__ == "__main__":
    cfg = Config()
    prepare_dynamic_datasets(cfg)

    # Display the distribution of y_test
    y_test = np.load(os.path.join(cfg.PROCESSED_DYNAMIC_DIR, "y_test.npy"))
    unique, counts = np.unique(y_test, return_counts=True)
    print("✔️ y_test distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")


