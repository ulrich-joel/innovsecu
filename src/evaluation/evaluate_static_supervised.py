# src/evaluation/evaluate_static_supervised.py
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from config.config import Config

config = Config()

# Load test features and true labels
y_test_path = os.path.join(config.PROCESSED_STATIC_DIR, 'y_test.npy')  # path to test labels
x_test = np.load(config.IF_KMEANS_TEST_FILE)  # test features
y_test = np.load(y_test_path)  # test labels

print("=== ⚡️ Isolation Forest Evaluation ===")
isolation_path = config.ISOLATION_FOREST_MODEL_PATH
isolation_forest = joblib.load(isolation_path)
# Isolation Forest predicts -1 for anomalies, convert to 1 for evaluation
y_pred_if = (isolation_forest.predict(x_test) == -1).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_if))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_if))

try:
    auc_if = roc_auc_score(y_test, y_pred_if)
    print(f"ROC AUC: {auc_if:.4f}")
except ValueError:
    print("ROC AUC: Cannot be computed (only one class present in y_test)")

print("\n=== ⚡️ KMeans Evaluation ===")
kmeans_path = config.KMEANS_MODEL_PATH
kmeans = joblib.load(kmeans_path)

# Calculate distances between each point and its assigned cluster center
distances = np.linalg.norm(x_test - kmeans.cluster_centers_[kmeans.predict(x_test)], axis=1)
threshold = np.percentile(distances, 95)  # arbitrary threshold at 95th percentile
# Points with distance above threshold are considered anomalies
y_pred_kmeans = (distances > threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_kmeans))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_kmeans))

try:
    auc_km = roc_auc_score(y_test, y_pred_kmeans)
    print(f"ROC AUC: {auc_km:.4f}")
except ValueError:
    print("ROC AUC: Cannot be computed")
