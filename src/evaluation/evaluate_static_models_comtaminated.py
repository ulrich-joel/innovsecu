# src/evaluation/evaluate_static_models_comtaminated.py
"""
====================================================================================
Dynamic Evaluation Script for Contaminated Static Anomaly Detection Models
====================================================================================

This script automatically evaluates all Isolation Forest models stored in the 
configured models directory. Each model is assumed to use a different contamination 
level and is named following the pattern: `isolation_forest_<contamination>.pkl`.

For each model:
    - Predictions and anomaly scores are generated.
    - Evaluation metrics (precision, recall, F1-score, AUC) are computed.
    - Confusion matrices and visual performance plots are created.

Visualizations:
    - ROC and Precision-Recall curves (4 models in one plot)
    - Confusion matrices (with TP/FP/FN/TN info per subplot)

------------------------------------------------------------------------------------

Usage:
    - Ensure that:
        - Test data is available as `x_test.npy` and `y_test.npy`
        - Trained models are placed in `models/` directory
    - File names for models should follow the pattern: `isolation_forest_<x>.pkl`

Outputs:
    - `isolation_forest_curves_4in1.png`
    - `isolation_forest_confusion_4in1.png`
    - Printed performance table

------------------------------------------------------------------------------------

Author  : Ngueyep Ulrich
Date    : 2025-10-14
Version : 1.0

====================================================================================
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay,
    confusion_matrix, ConfusionMatrixDisplay
)
from config.config import Config
import pandas as pd

# === Load configuration ===
config = Config()

# === Load test data ===
X_test = np.load(config.X_TEST_FILE)
y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)


def evaluate_model(name, y_true, y_pred, y_scores=None):
    """
    Evaluate a model using standard classification metrics.

    Args:
        name (str): Name of the model.
        y_true (np.ndarray): Ground truth binary labels (0 = normal, 1 = anomaly).
        y_pred (np.ndarray): Predicted binary labels.
        y_scores (np.ndarray or None): Anomaly scores or probabilities (used for AUC).

    Returns:
        dict: Evaluation results containing metrics and predictions.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores if y_scores is not None else y_pred)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n== {name} ==\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    return {
        "name": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "confusion_matrix": cm
    }


# === Evaluate all saved Isolation Forest models ===
results = []
print(" Evaluating Isolation Forest models:")

model_dir = config.MODELS_DIR_STATIC_TRAIN # Directory containing the models

for filename in os.listdir(model_dir):
    if filename.startswith("isolation_forest_") and filename.endswith(".pkl"):
        contamination = filename.split("_")[-1].replace(".pkl", "")
        model_path = os.path.join(model_dir, filename)
        model = joblib.load(model_path)

        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == -1, 1, 0)

        try:
            y_scores = -model.decision_function(X_test)  # higher = more anomalous
        except Exception:
            y_scores = None

        results.append(evaluate_model(f"IF_cont={contamination}", y_test, y_pred, y_scores))



print("\n Generating visualizations...")

output_dir = config.EVALUATION_DIR
os.makedirs(output_dir, exist_ok=True)

# Filter Isolation Forest models and sort by contamination level
if_models = [r for r in results if r['name'].startswith('IF_cont=')]
if_models_sorted = sorted(if_models, key=lambda x: float(x['name'].split('=')[1]))

# Select first 4 contamination levels (e.g., 1%, 5%, 10%, 15%)
selected_models = if_models_sorted[:4]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# === FIGURE : ROC & PR Curves ===
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

for i, (model, color) in enumerate(zip(selected_models, colors)):
    row, col = divmod(i, 2)
    scores = model['y_scores'] if model['y_scores'] is not None else model['y_pred']

    RocCurveDisplay.from_predictions(
        y_test,
        scores,
        ax=axes1[row, col],
        name=f"ROC (AUC: {model['auc']:.3f})",
        color=color,
        linestyle='-',
        linewidth=2
    )

    PrecisionRecallDisplay.from_predictions(
        y_test,
        scores,
        ax=axes1[row, col],
        name=f"PR (F1: {model['f1']:.3f})",
        color=color,
        linestyle='--',
        linewidth=2
    )

    axes1[row, col].plot([0, 1], [0, 1], 'k:', alpha=0.3, label='Random Classifier')

    axes1[row, col].set_title(
        f"{model['name']}\nPrecision: {model['precision']:.3f}, Recall: {model['recall']:.3f}",
        fontsize=12, fontweight='bold'
    )
    axes1[row, col].legend(loc='lower right', fontsize=9)
    axes1[row, col].grid(True, alpha=0.3)
    axes1[row, col].set_ylabel('TPR / Precision')

    if row == 1:
        axes1[row, col].set_xlabel('Recall')

plt.tight_layout()

roc_path = os.path.join(output_dir, 'isolation_forest_curves_4in1.png')
fig1.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"✅ Saved ROC and PR curves to: {roc_path}")


# === FIGURE : Matrices de confusion ===
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

for i, model in enumerate(selected_models):
    row, col = divmod(i, 2)
    cm = model['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=axes2[row, col], cmap=plt.cm.Blues, colorbar=False)
    axes2[row, col].set_title(model['name'], fontsize=12, fontweight='bold')

plt.tight_layout()

conf_mat_path = os.path.join(output_dir, 'isolation_forest_confusion_4in1.png')
fig2.savefig(conf_mat_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"✅ Saved confusion matrices to: {conf_mat_path}")

# === Print performance summary ===
if results:
    df = pd.DataFrame(results)[["name", "precision", "recall", "f1", "auc"]]
    print("\n Model Performance Summary:")
    print(df.to_string(index=False))
else:
    print(f" No Isolation Forest models were found in the directory: {model_dir}")
