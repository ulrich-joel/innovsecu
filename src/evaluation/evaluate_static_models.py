import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from config.config import Config

config = Config()

# --- Load test data ---
X_test = np.load(config.X_TEST_FILE)
y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)

# --- Helper to evaluate a model and display metrics ---
def evaluate_model(name, y_true, y_pred, y_scores=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores if y_scores is not None else y_pred)
    except ValueError:
        auc = float('nan')
    
    print(f"\n== {name} ==\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    
    RocCurveDisplay.from_predictions(
        y_true, y_scores if y_scores is not None else y_pred, name=f"{name} ROC"
    )
    PrecisionRecallDisplay.from_predictions(
        y_true, y_scores if y_scores is not None else y_pred, name=f"{name} PR"
    )
    plt.show()
    
    return {"name": name, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# --- Evaluate all Isolation Forest models ---
results = []
print("🔍 Evaluating Isolation Forest models:")

for filename in os.listdir(config.MODELS_DIR):
    if filename.startswith("isolation_forest_") and filename.endswith(".pkl"):
        contamination = filename.split("_")[-1].replace(".pkl", "")
        model = joblib.load(os.path.join(config.MODELS_DIR, filename))
        
        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == -1, 1, 0)
        
        try:
            y_scores = -model.decision_function(X_test)  # higher = more anomalous
        except:
            y_scores = None
        
        results.append(evaluate_model(f"IF_cont={contamination}", y_test, y_pred, y_scores))

# --- Evaluate KMeans model ---
print("\n🔍 Evaluating KMeans model:")
kmeans = joblib.load(config.KMEANS_MODEL_PATH)
cluster_labels = kmeans.predict(X_test)
dists = np.linalg.norm(X_test - kmeans.cluster_centers_[cluster_labels], axis=1)
threshold = np.percentile(dists, 95)
y_pred_kmeans = (dists > threshold).astype(int)

results.append(evaluate_model("KMeans (95th percentile)", y_test, y_pred_kmeans, dists))

# --- Display Summary ---
import pandas as pd
df = pd.DataFrame(results)[["name", "precision", "recall", "f1", "auc"]]
print("\n📊 Model Performance Summary:")
print(df.to_string(index=False))

# import numpy as np
# import joblib
# from config.config import Config
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     RocCurveDisplay,
#     PrecisionRecallDisplay,
# )
# import matplotlib.pyplot as plt

# # Load configuration
# config = Config()

# # Load test data and labels
# X_test = np.load(config.X_TEST_FILE)
# y_test = np.load(config.Y_TEST_FILE)

# # Sanity check
# assert len(X_test) == len(y_test), f"Shape mismatch: X_test={len(X_test)}, y_test={len(y_test)}"

# # Ensure y_test is binary (0 = normal, 1 = anomaly)
# y_test = (y_test > 0).astype(int)

# # Load trained models
# isolation_forest = joblib.load(config.ISOLATION_FOREST_MODEL_PATH)
# kmeans = joblib.load(config.KMEANS_MODEL_PATH)

# ### === Isolation Forest Evaluation === ###
# print("\n=== Isolation Forest Evaluation ===")
# if_raw_preds = isolation_forest.predict(X_test)  # -1 = anomaly, 1 = normal
# if_preds = np.where(if_raw_preds == -1, 1, 0)

# print(f"Detected anomalies: {np.sum(if_preds)} / {len(if_preds)}")
# print("Precision:", precision_score(y_test, if_preds))
# print("Recall:", recall_score(y_test, if_preds))
# print("F1 Score:", f1_score(y_test, if_preds))

# try:
#     print("AUC:", roc_auc_score(y_test, if_preds))
# except ValueError:
#     print("⚠️ AUC cannot be computed (only one class in y_test)")

# print("\nClassification Report:\n", classification_report(y_test, if_preds))
# print("Confusion Matrix:\n", confusion_matrix(y_test, if_preds))

# # Plot curves
# RocCurveDisplay.from_predictions(y_test, if_preds)
# plt.title("ROC Curve - Isolation Forest")
# plt.grid(True)
# plt.show()

# PrecisionRecallDisplay.from_predictions(y_test, if_preds)
# plt.title("Precision-Recall Curve - Isolation Forest")
# plt.grid(True)
# plt.show()


# ### === KMeans Evaluation === ###
# print("\n=== KMeans Evaluation ===")
# kmeans_labels = kmeans.predict(X_test)
# distances = np.linalg.norm(X_test - kmeans.cluster_centers_[kmeans_labels], axis=1)
# threshold = np.percentile(distances, 98)
# kmeans_preds = (distances > threshold).astype(int)

# print(f"Detected anomalies (threshold @95%): {np.sum(kmeans_preds)} / {len(kmeans_preds)}")
# print("Precision:", precision_score(y_test, kmeans_preds))
# print("Recall:", recall_score(y_test, kmeans_preds))
# print("F1 Score:", f1_score(y_test, kmeans_preds))

# try:
#     print("AUC:", roc_auc_score(y_test, kmeans_preds))
# except ValueError:
#     print("⚠️ AUC cannot be computed (only one class in y_test)")

# print("\nClassification Report:\n", classification_report(y_test, kmeans_preds))
# print("Confusion Matrix:\n", confusion_matrix(y_test, kmeans_preds))

# # Plot ROC and PR curves based on distance scores
# RocCurveDisplay.from_predictions(y_test, distances)
# plt.title("ROC Curve - KMeans (Distance Scores)")
# plt.grid(True)
# plt.show()

# PrecisionRecallDisplay.from_predictions(y_test, distances)
# plt.title("Precision-Recall Curve - KMeans (Distance Scores)")
# plt.grid(True)
# plt.show()