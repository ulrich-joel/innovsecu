# src/detection/train_static_supervised.py
"""
====================================================================================
 Train Meta-Classifiers Model Training Script
====================================================================================

Train meta-classifiers using features derived from ensemble models such as 
Isolation Forest and KMeans clustering. The trained meta-model selects the best 
classifier based on validation performance.
====================================================================================

Author: Ngueyep Ulrich
Date: 2025-10-13
====================================================================================
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config.config import Config
from src.logger import get_logger

logger = get_logger("train_meta_models")

# === Config & Paths ===
config = Config()
DATA_DIR = config.PROCESSED_STATIC_DIR
MODELS_DIR_STATIC_TRAIN = config.MODELS_DIR_STATIC_TRAIN
CONTAMINATIONS = [1, 5, 10, 15]

logger.info(" Loading static data...")
X_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "x_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

logger.info(" Loading models...")
kmeans_model = joblib.load(os.path.join(MODELS_DIR_STATIC_TRAIN, "kmeans.pkl"))
if_models = {
    cont: joblib.load(os.path.join(MODELS_DIR_STATIC_TRAIN, f"isolation_forest_{cont}.pkl"))
    for cont in CONTAMINATIONS
}

def get_meta_features(X):
    """
    Generate meta-features for the given input by combining the decision scores
    from Isolation Forests and the minimum distance from KMeans centroids.

    Args:
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Meta-feature matrix of shape (n_samples, n_models + 1).
    """
    features = [model.decision_function(X) for model in if_models.values()]
    kmeans_dist = kmeans_model.transform(X).min(axis=1)
    features.append(kmeans_dist)
    return np.vstack(features).T

logger.info(" Building meta-features...")
X_train_meta = get_meta_features(X_train)
X_val_meta = get_meta_features(X_val)
X_test_meta = get_meta_features(X_test)

logger.info(" Normalizing features...")
scaler = StandardScaler()
X_train_meta = scaler.fit_transform(X_train_meta)
X_val_meta = scaler.transform(X_val_meta)
X_test_meta = scaler.transform(X_test_meta)

# Define candidate classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
}

results = []

logger.info(" Training meta models...")
for name, clf in models.items():
    logger.info(f"Training {name}...")
    clf.fit(X_train_meta, y_train)
    f1_score = clf.score(X_val_meta, y_val)
    results.append({"Model": name, "ModelObject": clf, "ValScore": f1_score})
    logger.info(f"{name} validation score (approx.): {f1_score:.4f}")

# Save all trained meta-models individually
for result in results:
    model_name = result["Model"].replace(" ", "_").replace("(", "").replace(")", "").lower()
    model_path = os.path.join(MODELS_DIR_STATIC_TRAIN, f"meta_model_{model_name}.pkl")
    joblib.dump(result["ModelObject"], model_path)
    logger.info(f" Saved {result['Model']} to {model_path}")

# Still save the best model for quick access
best = max(results, key=lambda x: x["ValScore"])
best_model_path = os.path.join(MODELS_DIR_STATIC_TRAIN, "meta_best_model.pkl")
joblib.dump(best["ModelObject"], best_model_path)
joblib.dump(scaler, os.path.join(MODELS_DIR_STATIC_TRAIN, "meta_scaler.pkl"))
logger.info(f" Best model saved: {best['Model']} at {best_model_path}")



# # src/preprocessing/train_get_meta_model.py
# """
# train_get_meta_model.py

# This script is responsible for training meta-classifiers using features derived from Isolation Forest and KMeans models.
# It includes the following steps:
# 1. Load preprocessed static datasets (train, validation, and test splits).
# 2. Load trained Isolation Forest and KMeans models.
# 3. Generate meta-features using the decision scores of the Isolation Forest models and the distances from the KMeans model.
# 4. Train meta-classifiers (Logistic Regression, Random Forest, and SVM) using the generated meta-features.
# 5. Evaluate the meta-classifiers on the train, validation, and test datasets.
# 6. Visualize and save feature importance for each meta-classifier.
# 7. Save the best-performing meta-classifier and its scaler.

# Dependencies:
# - NumPy and Pandas for data manipulation.
# - Scikit-learn for preprocessing, model training, and evaluation.
# - Joblib for saving and loading models.
# - Matplotlib for plotting feature importance.
# - Config module for project-specific configurations.

# Outputs:
# - Meta-features saved as `.npy` files.
# - Trained meta-classifiers saved in the `MODELS_DIR`.
# - Feature importance plots saved as `.png` files.
# - Best-performing meta-classifier and its scaler saved in the `MODELS_DIR`.

# Author: Ngueyep Ulrich
# """

# import os
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance
# from config.config import Config

# # === Initialisation config ===
# config = Config()
# PROCESSED_STATIC_DIR = config.PROCESSED_STATIC_DIR
# MODELS_DIR = config.MODELS_DIR
# CONTAMINATIONS = [10]

# # === Chargement des données ===
# X_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_train.npy"))
# y_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_train.npy"))
# X_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_val.npy"))
# y_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_val.npy"))
# X_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_test.npy"))
# y_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_test.npy"))

# # === Chargement des modèles ===
# kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
# if_models = {
#     cont: joblib.load(os.path.join(MODELS_DIR, f"isolation_forest_{cont}.pkl"))
#     for cont in CONTAMINATIONS
# }

# # === Fonction de création de méta-features ===
# def get_meta_features(X):
#     features = []
#     for cont, model in if_models.items():
#         if_score = model.decision_function(X)
#         features.append(if_score)
    
#     try:
#         kmeans_dist = kmeans_model.transform(X).min(axis=1)
#         features.append(kmeans_dist)
#     except ValueError as e:
#         raise ValueError(f"Erreur de dimensions dans KMeans.transform(): {e}\n"
#                          f"KMeans attend {kmeans_model.n_features_in_} features, mais X en a {X.shape[1]}.")

#     return np.vstack(features).T

# # === Construction des méta-features ===
# X_train_meta = get_meta_features(X_train)
# X_val_meta = get_meta_features(X_val)
# X_test_meta = get_meta_features(X_test)

# # === Normalisation ===
# scaler = StandardScaler()
# X_train_meta = scaler.fit_transform(X_train_meta)
# X_val_meta = scaler.transform(X_val_meta)
# X_test_meta = scaler.transform(X_test_meta)

# # === Évaluation ===
# def evaluate_model(model, X, y, name=""):
#     preds = model.predict(X)
#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(X)[:, 1]
#     else:
#         probs = model.decision_function(X)
#     precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
#     auc = roc_auc_score(y, probs)
#     print(f"== {name} ==")
#     print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}\n")
#     return precision, recall, f1, auc

# # === Entraînement des méta-classifieurs ===
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
# }

# results = []
# print("🔍 Évaluation des modèles méta :\n")
# feature_names = [f"IF_{c}%" for c in CONTAMINATIONS] + ["KMeans"]

# for name, clf in models.items():
#     clf.fit(X_train_meta, y_train)
#     print(f"🔹 {name}")
#     train_scores = evaluate_model(clf, X_train_meta, y_train, "Train")
#     val_scores = evaluate_model(clf, X_val_meta, y_val, "Validation")
#     test_scores = evaluate_model(clf, X_test_meta, y_test, "Test")
#     results.append({
#         "Model": name,
#         "Train F1": train_scores[2],
#         "Validation F1": val_scores[2],
#         "Test F1": test_scores[2]
#     })

#     # === Importance des features ===
#     plt.figure(figsize=(8, 4))
#     if name == "Logistic Regression":
#         importances = np.abs(clf.coef_[0])
#         plt.title("Logistic Regression - Feature Importance")
#     elif name == "Random Forest":
#         importances = clf.feature_importances_
#         plt.title("Random Forest - Feature Importance")
#     elif name == "SVM (Linear)":
#         print("🌀 Calcul de l'importance par permutation pour SVM...")
#         perm = permutation_importance(clf, X_val_meta, y_val, n_repeats=20, random_state=42)
#         importances = perm.importances_mean
#         plt.title("SVM - Permutation Feature Importance")

#     plt.bar(feature_names, importances)
#     plt.ylabel("Importance")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plot_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
#     plt.savefig(os.path.join(MODELS_DIR, f"{plot_name}_feature_importance.png"))
#     plt.show()
#     print(f"📈 Graphique sauvegardé : {plot_name}_feature_importance.png\n")

# # === Résumé des performances ===
# df_results = pd.DataFrame(results)
# print("\n📊 Résumé des F1 scores :")
# print(df_results.to_string(index=False))

# # === Sauvegarde du meilleur modèle ===
# best_model_name = df_results.sort_values("Test F1", ascending=False).iloc[0]["Model"]
# best_model = models[best_model_name]
# joblib.dump(best_model, os.path.join(MODELS_DIR, "meta_best_model.pkl"))
# joblib.dump(scaler, os.path.join(MODELS_DIR, "meta_scaler.pkl"))
# print(f"\n✅ Meilleur modèle méta : {best_model_name} sauvegardé.")

# import os
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance
# import pandas as pd
# from src.detection.model_trainer import ModelTrainer  # ✅ et plus src.detection.vae_model


# # === Paths ===
# PROCESSED_STATIC_DIR = "data/processed/processed_static"
# MODELS_DIR = "output/models"
# CONTAMINATIONS = [10]

# # === Load static data ===
# X_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_train.npy"))
# y_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_train.npy"))
# X_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_val.npy"))
# y_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_val.npy"))
# X_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_test.npy"))
# y_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_test.npy"))

# # === Load KMeans model ===
# kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))

# # === Load all Isolation Forest models ===
# if_models = {
#     cont: joblib.load(os.path.join(MODELS_DIR, f"isolation_forest_{cont}.pkl"))
#     for cont in CONTAMINATIONS
# }

# # === Build meta features ===
# def get_meta_features(X):
#     features = []
#     for cont, model in if_models.items():
#         if_score = model.decision_function(X)
#         features.append(if_score)
#     # Add KMeans distance
#     kmeans_dist = kmeans_model.transform(X).min(axis=1)
#     features.append(kmeans_dist)
#     return np.vstack(features).T

# X_train_meta = get_meta_features(X_train)
# X_val_meta = get_meta_features(X_val)
# X_test_meta = get_meta_features(X_test)

# # === Normalize meta features ===
# scaler = StandardScaler()
# X_train_meta = scaler.fit_transform(X_train_meta)
# X_val_meta = scaler.transform(X_val_meta)
# X_test_meta = scaler.transform(X_test_meta)

# # === Evaluation helper ===
# def evaluate_model(model, X, y, name=""):
#     preds = model.predict(X)
#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(X)[:, 1]
#     else:
#         # For models without predict_proba (e.g. SVM), use decision_function
#         probs = model.decision_function(X)
#     precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
#     auc = roc_auc_score(y, probs)
#     print(f"== {name} ==")
#     print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}\n")
#     return precision, recall, f1, auc

# # === Train and evaluate classifiers ===
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
# }

# results = []

# print("🔍 Evaluating meta models:\n")

# for name, clf in models.items():
#     clf.fit(X_train_meta, y_train)
#     print(f"🔹 {name}")
#     train_scores = evaluate_model(clf, X_train_meta, y_train, "Train")
#     val_scores = evaluate_model(clf, X_val_meta, y_val, "Validation")
#     test_scores = evaluate_model(clf, X_test_meta, y_test, "Test")
#     results.append({
#         "Model": name,
#         "Train F1": train_scores[2],
#         "Validation F1": val_scores[2],
#         "Test F1": test_scores[2]
#     })

#     # === Feature importance ===
#     feature_names = [f"IF_{c}%" for c in CONTAMINATIONS] + ["KMeans"]

#     if name == "Logistic Regression":
#         importances = np.abs(clf.coef_[0])
#         plt.figure(figsize=(8,4))
#         plt.title("Logistic Regression - Feature Importance (absolute coef)")
#         plt.bar(feature_names, importances)
#         plt.ylabel("Absolute Coefficient")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(MODELS_DIR, "logistic_feature_importance.png"))
#         plt.show()
#         print("📈 Saved Logistic Regression feature importance plot.\n")

#     elif name == "Random Forest":
#         importances = clf.feature_importances_
#         plt.figure(figsize=(8,4))
#         plt.title("Random Forest - Feature Importance")
#         plt.bar(feature_names, importances)
#         plt.ylabel("Feature Importance")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(MODELS_DIR, "rf_feature_importance.png"))
#         plt.show()
#         print("📈 Saved Random Forest feature importance plot.\n")

#     elif name == "SVM (Linear)":
#         # Use permutation importance for SVM
#         print("Computing permutation importance for SVM...")
#         perm_importance = permutation_importance(clf, X_val_meta, y_val, n_repeats=20, random_state=42)
#         importances = perm_importance.importances_mean
#         plt.figure(figsize=(8,4))
#         plt.title("SVM (Linear) - Permutation Feature Importance")
#         plt.bar(feature_names, importances)
#         plt.ylabel("Mean Decrease in Score")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(MODELS_DIR, "svm_permutation_feature_importance.png"))
#         plt.show()
#         print("📈 Saved SVM permutation feature importance plot.\n")

# # === Summary of F1 scores ===
# df_results = pd.DataFrame(results)
# print("\n📊 Summary of F1 scores (Train / Validation / Test):")
# print(df_results.to_string(index=False))

# # === Save best model (by Test F1) ===
# best_model_name = df_results.sort_values("Test F1", ascending=False).iloc[0]["Model"]
# best_model = models[best_model_name]
# joblib.dump(best_model, os.path.join(MODELS_DIR, "meta_best_model.pkl"))
# joblib.dump(scaler, os.path.join(MODELS_DIR, "meta_scaler.pkl"))

# print(f"\n✅ Best meta model: {best_model_name} saved.")

# import os
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance

# # === Paths ===
# PROCESSED_STATIC_DIR = "data/processed/processed_static"
# MODELS_DIR = "output/models"
# CONTAMINATIONS = [1, 5, 10, 15]  # Percentages

# # === Load static data ===
# X_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_train.npy"))
# y_train = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_train.npy"))
# X_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_val.npy"))
# y_val = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_val.npy"))
# X_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "x_test.npy"))
# y_test = np.load(os.path.join(PROCESSED_STATIC_DIR, "y_test.npy"))

# # === Load KMeans model ===
# kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))

# # === Load all Isolation Forest models ===
# if_models = {
#     cont: joblib.load(os.path.join(MODELS_DIR, f"isolation_forest_{cont}.pkl"))
#     for cont in CONTAMINATIONS
# }

# # === Build meta features ===
# def get_meta_features(X):
#     features = []
#     for cont, model in if_models.items():
#         if_score = model.decision_function(X)
#         features.append(if_score)
#     # Add KMeans distance
#     kmeans_dist = kmeans_model.transform(X).min(axis=1)
#     features.append(kmeans_dist)
#     return np.vstack(features).T

# X_train_meta = get_meta_features(X_train)
# X_val_meta = get_meta_features(X_val)
# X_test_meta = get_meta_features(X_test)

# # === Normalize meta features ===
# scaler = StandardScaler()
# X_train_meta = scaler.fit_transform(X_train_meta)
# X_val_meta = scaler.transform(X_val_meta)
# X_test_meta = scaler.transform(X_test_meta)

# # === Evaluation helper ===
# def evaluate_model(model, X, y, name=""):
#     preds = model.predict(X)
#     probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
#     precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
#     auc = roc_auc_score(y, probs)
#     print(f"== {name} ==")
#     print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}\n")
#     return precision, recall, f1, auc

# # === Train and evaluate classifiers ===
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
# }

# results = {}

# print("🔍 Evaluating meta models:\n")

# for name, clf in models.items():
#     clf.fit(X_train_meta, y_train)
#     print(f"🔹 {name}")
#     evaluate_model(clf, X_train_meta, y_train, "Train")
#     evaluate_model(clf, X_val_meta, y_val, "Validation")
#     results[name] = evaluate_model(clf, X_test_meta, y_test, "Test")

#     # === Feature importance for Logistic Regression and Random Forest ===
#     if name == "Logistic Regression":
#         plt.figure()
#         plt.title("Logistic Regression - Feature Importance")
#         plt.bar(range(X_train_meta.shape[1]), np.abs(clf.coef_[0]))
#         plt.xticks(range(X_train_meta.shape[1]), [f"IF_{c}" for c in CONTAMINATIONS] + ["KMeans"], rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(MODELS_DIR, "logistic_feature_importance.png"))
#         print("📈 Saved Logistic Regression feature importance plot.\n")

#     elif name == "Random Forest":
#         plt.figure()
#         plt.title("Random Forest - Feature Importance")
#         plt.bar(range(X_train_meta.shape[1]), clf.feature_importances_)
#         plt.xticks(range(X_train_meta.shape[1]), [f"IF_{c}" for c in CONTAMINATIONS] + ["KMeans"], rotation=45)
#         plt.tight_layout()
#         plt.savefig(os.path.join(MODELS_DIR, "rf_feature_importance.png"))
#         print("📈 Saved Random Forest feature importance plot.\n")

# # === Save best model (optional) ===
# best_model_name = max(results.items(), key=lambda x: x[1][2])[0]  # By F1 score
# best_model = models[best_model_name]
# joblib.dump(best_model, os.path.join(MODELS_DIR, "meta_best_model.pkl"))
# joblib.dump(scaler, os.path.join(MODELS_DIR, "meta_scaler.pkl"))

# print(f"✅ Best meta model: {best_model_name} saved.")
