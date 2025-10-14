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
print("🔍 Evaluating Isolation Forest models:")

for filename in os.listdir(config.MODELS_DIR):
    if filename.startswith("isolation_forest_") and filename.endswith(".pkl"):
        contamination = filename.split("_")[-1].replace(".pkl", "")
        model_path = os.path.join(config.MODELS_DIR, filename)
        model = joblib.load(model_path)

        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == -1, 1, 0)

        try:
            y_scores = -model.decision_function(X_test)  # higher = more anomalous
        except Exception:
            y_scores = None

        results.append(evaluate_model(f"IF_cont={contamination}", y_test, y_pred, y_scores))


# === Visualization: ROC & Precision-Recall Curves ===
print("\n📊 Generating visualizations...")

# Filter Isolation Forest models and sort by contamination level
if_models = [r for r in results if r['name'].startswith('IF_cont=')]
if_models_sorted = sorted(if_models, key=lambda x: float(x['name'].split('=')[1]))

# Select first 4 contamination levels (e.g., 1%, 5%, 10%, 15%)
selected_models = if_models_sorted[:4]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# === FIGURE 1: ROC & PR Curves ===
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

for i, (model, color) in enumerate(zip(selected_models, colors)):
    row, col = divmod(i, 2)
    scores = model['y_scores'] if model['y_scores'] is not None else model['y_pred']

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test,
        scores,
        ax=axes1[row, col],
        name=f"ROC (AUC: {model['auc']:.3f})",
        color=color,
        linestyle='-',
        linewidth=2
    )

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test,
        scores,
        ax=axes1[row, col],
        name=f"PR (F1: {model['f1']:.3f})",
        color=color,
        linestyle='--',
        linewidth=2
    )

    # Random classifier line
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
plt.savefig('isolation_forest_curves_4in1.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ ROC and Precision-Recall curves saved as 'isolation_forest_curves_4in1.png'")


# === FIGURE 2: Confusion Matrices ===
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

for i, model in enumerate(selected_models):
    row, col = divmod(i, 2)
    cm = model['confusion_matrix']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=axes2[row, col], cmap='Blues', values_format='d')

    axes2[row, col].set_title(
        f"{model['name']}\nTP: {cm[1,1]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TN: {cm[0,0]}",
        fontsize=12, fontweight='bold'
    )

    metrics_text = (
        f"Precision: {model['precision']:.3f}\n"
        f"Recall: {model['recall']:.3f}\n"
        f"F1: {model['f1']:.3f}"
    )

    axes2[row, col].text(
        0.95, 0.05, metrics_text,
        transform=axes2[row, col].transAxes,
        ha='right', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

plt.tight_layout()
plt.savefig('isolation_forest_confusion_4in1.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved as 'isolation_forest_confusion_4in1.png'")


# === Print performance summary ===
df = pd.DataFrame(results)[["name", "precision", "recall", "f1", "auc"]]
print("\n📊 Model Performance Summary:")
print(df.to_string(index=False))