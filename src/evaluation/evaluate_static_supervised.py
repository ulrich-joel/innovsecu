# src/evaluation/evaluate_static_supervised.py
"""
===============================================================================
 Evaluate Model Training Script
===============================================================================

Evaluate multiple meta-classifiers on test data using metrics 
such as precision, recall, F1-score, and AUC. Generates ROC curve,
confusion matrix, and feature importance for each.

===============================================================================

Author: Ngueyep Ulrich
Date: 2025-10-13
===============================================================================
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from config.config import Config
from src.logger import get_logger

logger = get_logger("evaluate_meta_models")

# === Config ===
config = Config()
DATA_DIR = config.PROCESSED_STATIC_DIR
EVAL_DIR = config.EVALUATION_DIR
MODEL_DIR = config.MODELS_DIR_STATIC_TRAIN
CONTAMINATIONS = [1, 5, 10, 15]
feature_names = [f"IF_{c}%" for c in CONTAMINATIONS] + ["KMeans"]

# === Load test data and models ===
logger.info(" Loading test data and models...")
X_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
scaler = joblib.load(os.path.join(MODEL_DIR, "meta_scaler.pkl"))

model_files = {
    "Logistic Regression": "meta_model_logistic_regression.pkl",
    "Random Forest": "meta_model_random_forest.pkl",
    "SVM (Linear)": "meta_model_svm_linear.pkl"
}
models = {
    name: joblib.load(os.path.join(MODEL_DIR, fname))
    for name, fname in model_files.items()
}

kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
if_models = {
    cont: joblib.load(os.path.join(MODEL_DIR, f"isolation_forest_{cont}.pkl"))
    for cont in CONTAMINATIONS
}

# === Feature construction ===
def get_meta_features(X):
    features = [model.decision_function(X) for model in if_models.values()]
    kmeans_dist = kmeans_model.transform(X).min(axis=1)
    features.append(kmeans_dist)
    return np.vstack(features).T

X_test_meta = scaler.transform(get_meta_features(X_test))

# === Evaluation function ===
def evaluate_model(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
    auc = roc_auc_score(y, probs)
    return preds, probs, precision, recall, f1, auc

# === Couleurs fixes pour chaque modèle ===
color_map = {
    "Logistic Regression": "blue",
    "Random Forest": "green",
    "SVM (Linear)": "red"
}

# === Evaluation, plots individuels, et collecte confusion matrices ===
conf_matrices = {}
roc_data = {}

for name, model in models.items():
    logger.info(f"🔍 Evaluating {name}...")
    preds, probs, precision, recall, f1, auc = evaluate_model(model, X_test_meta, y_test)
    logger.info(f"{name} - Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    
    # ROC Curve individuelle
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color_map.get(name, None))
    plt.plot([0, 1], [0, 1], 'k--', label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(EVAL_DIR, f"roc_curve_{name.replace(' ', '_').lower()}.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"ROC curve saved at: {roc_path}")
    
    # Stocker les données ROC pour fusion
    roc_data[name] = (fpr, tpr, auc)

    # Confusion Matrix individuelle
    cm = confusion_matrix(y_test, preds)
    conf_matrices[name] = cm
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    cm_path = os.path.join(EVAL_DIR, f"conf_matrix_{name.replace(' ', '_').lower()}.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved at: {cm_path}")

    # Feature Importance individuelle
    plt.figure(figsize=(8, 4))
    if hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
        plt.title(f"Feature Importance ({name})")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.title(f"Feature Importance ({name})")
    else:
        logger.info(f"Using permutation importance for {name}...")
        perm = permutation_importance(model, X_test_meta, y_test, n_repeats=20, random_state=42)
        importances = perm.importances_mean
        plt.title(f"Feature Importance ({name} - Permutation)")
    
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    importance_path = os.path.join(EVAL_DIR, f"feature_importance_{name.replace(' ', '_').lower()}.png")
    plt.savefig(importance_path)
    plt.close()
    logger.info(f"Feature importance plot saved at: {importance_path}")

# === Combined Confusion Matrix Plot ===
fig, axes = plt.subplots(1, len(conf_matrices), figsize=(5 * len(conf_matrices), 4))

if len(conf_matrices) == 1:
    axes = [axes]  # handle single plot case

for ax, (name, cm) in zip(axes, conf_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues)  # colorbar visible for better contrast
    ax.set_title(name)

plt.suptitle("Matrice de confusion - Tous les modèles")
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt_path = os.path.join(EVAL_DIR, "all_confusion_matrices.png")
plt.savefig(plt_path)
plt.show()
logger.info(f"Combined confusion matrices saved at: {plt_path}")

# === Combined ROC Curve Plot ===
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color_map.get(name, None))

plt.plot([0, 1], [0, 1], 'k--', label="Random chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tous les modèles")
plt.legend(loc="lower right")
plt.tight_layout()
roc_all_path = os.path.join(EVAL_DIR, "roc_curve_all_models.png")
plt.savefig(roc_all_path)
plt.show()
logger.info(f"ROC curve (all models) saved at: {roc_all_path}")

# === Comparative Feature Importance Plot ===
logger.info("Plotting comparative feature importance...")

importances_dict = {}

for name, model in models.items():
    if hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.info(f"Using permutation importance for {name}...")
        perm = permutation_importance(model, X_test_meta, y_test, n_repeats=20, random_state=42)
        importances = perm.importances_mean
    importances_dict[name] = importances

# Tracer un graphique comparatif côte à côte
fig, axs = plt.subplots(1, len(importances_dict), figsize=(15, 4), sharey=True)
colors = ["skyblue", "lightgreen", "lightcoral"]

for idx, (name, importances) in enumerate(importances_dict.items()):
    axs[idx].bar(feature_names, importances, color=colors[idx % len(colors)])
    axs[idx].set_title(name)
    axs[idx].set_xticks(range(len(feature_names))) 
    axs[idx].set_xticklabels(feature_names, rotation=45)
    axs[idx].set_ylabel("Importance")

plt.suptitle("Comparaison des importances des caractéristiques (par modèle)")
plt.tight_layout()

# Sauvegarde du graphique comparatif
feature_plot_path = os.path.join(EVAL_DIR, "comparative_feature_importance.png")
plt.savefig(feature_plot_path)
plt.show()
logger.info(f"Comparative feature importance plot saved at: {feature_plot_path}")
