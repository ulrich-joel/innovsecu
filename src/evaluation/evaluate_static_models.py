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

# Initialize configuration
config = Config()

# --- Load test data ---
X_test = np.load(config.X_TEST_FILE)
y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)

# --- Helper to evaluate a model and return metrics ---
def evaluate_model(name, y_true, y_pred, y_scores=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_scores if y_scores is not None else y_pred)
    except ValueError:
        auc = float('nan')
    
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

# --- Create visualizations ---
print("\n📊 Generating visualizations...")

# Select all 4 Isolation Forest contamination levels
if_models = [r for r in results if r['name'].startswith('IF_cont=')]
if_models_sorted = sorted(if_models, key=lambda x: float(x['name'].split('=')[1]))

# Take the 4 contamination levels
selected_models = if_models_sorted[:4]  # 1%, 5%, 10%, 15%

# Define distinct colors for each contamination level
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# === FIGURE 1: ROC and Precision-Recall Curves ===
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

for i, (model, color) in enumerate(zip(selected_models, colors)):
    row = i // 2
    col = i % 2
    
    # Get scores for plotting
    scores = model['y_scores'] if model['y_scores'] is not None else model['y_pred']
    
    # Plot ROC curve (solid line)
    RocCurveDisplay.from_predictions(
        y_test, 
        scores, 
        ax=axes1[row, col],
        name=f"ROC (AUC: {model['auc']:.3f})",
        color=color,
        linestyle='-',
        linewidth=2
    )
    
    # Plot Precision-Recall curve (dashed line)
    PrecisionRecallDisplay.from_predictions(
        y_test, 
        scores, 
        ax=axes1[row, col],
        name=f"PR (F1: {model['f1']:.3f})",
        color=color,
        linestyle='--',
        linewidth=2
    )
    
    # Add diagonal line for ROC reference
    axes1[row, col].plot([0, 1], [0, 1], 'k:', alpha=0.3, label='Random Classifier')
    
    # Customize the subplot
    axes1[row, col].set_title(f"{model['name']}\nPrecision: {model['precision']:.3f}, Recall: {model['recall']:.3f}", 
                           fontsize=12, fontweight='bold', pad=10)
    axes1[row, col].legend(loc='lower right', fontsize=9)
    axes1[row, col].grid(True, alpha=0.3)
    axes1[row, col].set_ylabel('True Positive Rate / Precision')
    
    # Add right-side label for False Positive Rate
    axes1[row, col].text(1.02, 0.5, 'False Positive Rate', transform=axes1[row, col].transAxes, 
                       rotation=270, va='center', ha='left', fontsize=11)
    
    # Only show "Recall" label on bottom row
    if row == 1:  # Bottom row
        axes1[row, col].set_xlabel('Recall')
    else:  # Top row - no xlabel
        axes1[row, col].set_xlabel('')

# plt.suptitle('ROC and Precision-Recall Curves - Isolation Forest with Different Contamination Levels\n(Solid: ROC, Dashed: Precision-Recall)', 
#              fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('isolation_forest_curves_4in1.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ ROC and Precision-Recall curves 4-in-1 saved as 'isolation_forest_curves_4in1.png'")

# === FIGURE 2: Confusion Matrices ===
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

for i, model in enumerate(selected_models):
    row = i // 2
    col = i % 2
    
    # Plot confusion matrix
    cm = model['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=axes2[row, col], cmap='Blues', values_format='d')
    
    # Customize the subplot
    axes2[row, col].set_title(f"{model['name']}\nTP: {cm[1,1]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TN: {cm[0,0]}", 
                            fontsize=12, fontweight='bold', pad=10)
    
    # Add performance metrics to the plot
    metrics_text = f"Precision: {model['precision']:.3f}\nRecall: {model['recall']:.3f}\nF1: {model['f1']:.3f}"
    axes2[row, col].text(0.95, 0.05, metrics_text, transform=axes2[row, col].transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)

# plt.suptitle('Confusion Matrices - Isolation Forest with Different Contamination Levels', 
#              fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('isolation_forest_confusion_4in1.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices 4-in-1 saved as 'isolation_forest_confusion_4in1.png'")

# --- Display Summary ---
import pandas as pd
df = pd.DataFrame(results)[["name", "precision", "recall", "f1", "auc"]]
print("\n📊 Model Performance Summary:")
print(df.to_string(index=False))
