import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import os
from datetime import datetime

def plot_conf_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def evaluate_models(trainer, X_test, y_test):
    logger = logging.getLogger(__name__)

    # 🔒 Sécurité : vérification que X_test et y_test ont la même taille
    if len(X_test) != len(y_test):
        raise ValueError(f"❌ Mismatch entre X_test ({len(X_test)}) et y_test ({len(y_test)})")

    model = trainer.model.eval()
    os.makedirs("outputs/evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
        recon_x, _, _ = model(X_tensor)
        recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()

    # VAE
    vae_threshold = np.percentile(recon_errors, 95)
    vae_preds = (recon_errors > vae_threshold).astype(int)
    vae_auc = roc_auc_score(y_test, recon_errors)
    vae_prec = precision_score(y_test, vae_preds)
    vae_rec = recall_score(y_test, vae_preds)
    vae_f1 = f1_score(y_test, vae_preds)

    logger.info("\n🔍 [VAE] Reconstruction threshold = %.4f", vae_threshold)
    logger.info("[VAE] Classification Report:\n" + classification_report(y_test, vae_preds))
    logger.info("[VAE] AUC: %.4f", vae_auc)


    # Isolation Forest
    if_scores = -trainer.isolation_forest.decision_function(X_test)
    if_threshold = np.percentile(if_scores, 95)
    if_preds = (trainer.isolation_forest.predict(X_test) == -1).astype(int)
    if_auc = roc_auc_score(y_test, if_scores)
    if_prec = precision_score(y_test, if_preds)
    if_rec = recall_score(y_test, if_preds)
    if_f1 = f1_score(y_test, if_preds)

    logger.info("[Isolation Forest] Classification Report:\n" + classification_report(y_test, if_preds))
    logger.info("[Isolation Forest] AUC: %.4f", if_auc)

    # KMeans
    distances = trainer.kmeans.transform(X_test).min(axis=1)
    km_threshold = np.percentile(distances, 95)
    km_preds = (distances > km_threshold).astype(int)
    km_auc = roc_auc_score(y_test, distances)
    km_prec = precision_score(y_test, km_preds)
    km_rec = recall_score(y_test, km_preds)
    km_f1 = f1_score(y_test, km_preds)

    logger.info("\n🔍 [KMeans] Distance threshold = %.4f", km_threshold)
    logger.info("[KMeans] Classification Report:\n" + classification_report(y_test, km_preds))
    logger.info("[KMeans] AUC: %.4f", km_auc)

    # Résumé
    summary = pd.DataFrame([
        ["VAE", vae_auc, vae_prec, vae_rec, vae_f1],
        ["Isolation Forest", if_auc, if_prec, if_rec, if_f1],
        ["KMeans", km_auc, km_prec, km_rec, km_f1],
    ], columns=["Model", "AUC", "Precision", "Recall", "F1"])
    summary.to_csv(f"outputs/evaluation/auc_summary_{timestamp}.csv", index=False)
    logger.info("\n=== Résumé des AUCs ===\n" + summary.to_string(index=False))

    # ROC curves
    fpr_vae, tpr_vae, _ = roc_curve(y_test, recon_errors)
    fpr_if, tpr_if, _ = roc_curve(y_test, if_scores)
    fpr_km, tpr_km, _ = roc_curve(y_test, distances)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_vae, tpr_vae, label=f"VAE (AUC={vae_auc:.2f})")
    plt.plot(fpr_if, tpr_if, label=f"Isolation Forest (AUC={if_auc:.2f})")
    plt.plot(fpr_km, tpr_km, label=f"KMeans (AUC={km_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_path = f"outputs/evaluation/roc_comparison_{timestamp}.png"
    plt.savefig(roc_path)
    plt.close()

    # PR curves
    prec_vae, rec_vae, _ = precision_recall_curve(y_test, recon_errors)
    prec_if, rec_if, _ = precision_recall_curve(y_test, if_scores)
    prec_km, rec_km, _ = precision_recall_curve(y_test, distances)

    plt.figure(figsize=(10, 6))
    plt.plot(rec_vae, prec_vae, label="VAE")
    plt.plot(rec_if, prec_if, label="Isolation Forest")
    plt.plot(rec_km, prec_km, label="KMeans")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = f"outputs/evaluation/pr_comparison_{timestamp}.png"
    plt.savefig(pr_path)
    plt.close()

    # Matrices de confusion
    plot_conf_matrix(y_test, vae_preds, "Confusion Matrix - VAE", f"outputs/evaluation/cm_vae_{timestamp}.png")
    plot_conf_matrix(y_test, if_preds, "Confusion Matrix - Isolation Forest", f"outputs/evaluation/cm_if_{timestamp}.png")
    plot_conf_matrix(y_test, km_preds, "Confusion Matrix - KMeans", f"outputs/evaluation/cm_km_{timestamp}.png")

    # Rapport PDF
    with PdfPages(f"outputs/evaluation/eval_report_{timestamp}.pdf") as pdf:
        for path in [roc_path, pr_path,
                     f"outputs/evaluation/cm_vae_{timestamp}.png",
                     f"outputs/evaluation/cm_if_{timestamp}.png",
                     f"outputs/evaluation/cm_km_{timestamp}.png"]:
            img = plt.imread(path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig()
            plt.close()
    
    logger.info(f"[KMeans] AUC: {km_auc:.4f}")

    trainer.thresholds = {
        "vae": vae_threshold,
        "if": if_threshold,
        "kmeans": km_threshold
    }


    # Résumé
    logger.info("\n=== Résumé des AUCs ===")

def save_models(trainer, config):
    torch.save(trainer.model.state_dict(), config.BEST_VAE_MODEL_PATH)
    joblib.dump(trainer.isolation_forest, config.BEST_ISOFOREST_MODEL_PATH)
    joblib.dump(trainer.kmeans, config.BEST_KMEANS_MODEL_PATH)


# import logging
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
# import matplotlib.pyplot as plt
# import joblib
# import os
# from datetime import datetime


# def evaluate_models(trainer, X_test, y_test):
#     logger = logging.getLogger(__name__)
#     model = trainer.model.eval()

#     with torch.no_grad():
#         X_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
#         recon_x, _, _ = model(X_tensor)
#         recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()

#     # VAE thresholding
#     vae_threshold = np.percentile(recon_errors, 95)
#     vae_preds = (recon_errors > vae_threshold).astype(int)
#     vae_auc = roc_auc_score(y_test, recon_errors)

#     logger.info("\n🔍 [VAE] Reconstruction threshold = %.4f", vae_threshold)
#     logger.info("[VAE] Classification Report:\n" + classification_report(y_test, vae_preds))
#     logger.info("[VAE] AUC: %.4f", vae_auc)

#     # Isolation Forest
#     if_scores = -trainer.isolation_forest.decision_function(X_test)
#     if_preds = (trainer.isolation_forest.predict(X_test) == -1).astype(int)
#     if_auc = roc_auc_score(y_test, if_scores)

#     logger.info("[Isolation Forest] Classification Report:\n" + classification_report(y_test, if_preds))
#     logger.info("[Isolation Forest] AUC: %.4f", if_auc)

#     # KMeans
#     distances = trainer.kmeans.transform(X_test).min(axis=1)
#     km_threshold = np.percentile(distances, 95)
#     km_preds = (distances > km_threshold).astype(int)
#     km_auc = roc_auc_score(y_test, distances)

#     logger.info("\n🔍 [KMeans] Distance threshold = %.4f", km_threshold)
#     logger.info("[KMeans] Classification Report:\n" + classification_report(y_test, km_preds))
#     logger.info("[KMeans] AUC: %.4f", km_auc)

#     # Résumé au format tableau
#     summary = pd.DataFrame([
#         ["VAE", vae_auc],
#         ["Isolation Forest", if_auc],
#         ["KMeans", km_auc]
#     ], columns=["Model", "AUC"])

#     # 🧾 Sauvegarde du résumé AUC dans un fichier CSV
#     os.makedirs("outputs/evaluation", exist_ok=True)
#     summary.to_csv("outputs/evaluation/auc_summary.csv", index=False)


#     logger.info("\n=== Résumé des AUCs ===\n" + summary.to_string(index=False))

#     # Courbes ROC
#     fpr_vae, tpr_vae, _ = roc_curve(y_test, recon_errors)
#     fpr_if, tpr_if, _ = roc_curve(y_test, if_scores)
#     fpr_km, tpr_km, _ = roc_curve(y_test, distances)

#     # Courbes ROC
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr_vae, tpr_vae, label=f"VAE (AUC={vae_auc:.2f})")
#     plt.plot(fpr_if, tpr_if, label=f"Isolation Forest (AUC={if_auc:.2f})")
#     plt.plot(fpr_km, tpr_km, label=f"KMeans (AUC={km_auc:.2f})")
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curves")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f"outputs/evaluation/roc_comparison_{timestamp}.png")
#     plt.close()

#     # Courbes Precision-Recall
#     prec_vae, rec_vae, _ = precision_recall_curve(y_test, recon_errors)
#     prec_if, rec_if, _ = precision_recall_curve(y_test, if_scores)
#     prec_km, rec_km, _ = precision_recall_curve(y_test, distances)

#     plt.figure(figsize=(10, 6))
#     plt.plot(rec_vae, prec_vae, label="VAE")
#     plt.plot(rec_if, prec_if, label="Isolation Forest")
#     plt.plot(rec_km, prec_km, label="KMeans")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curves")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"outputs/evaluation/pr_comparison_{timestamp}.png")  # ✅ Correctement placé ici
#     plt.close()


# def save_models(trainer, config):
#     torch.save(trainer.model.state_dict(), config.BEST_VAE_MODEL_PATH)
#     joblib.dump(trainer.isolation_forest, config.BEST_ISOFOREST_MODEL_PATH)
#     joblib.dump(trainer.kmeans, config.BEST_KMEANS_MODEL_PATH)
    


# # src/evaluation/evaluate_dynamic_supervised.py
# import os
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import joblib
# from config.config import Config

# config = Config()

# # Load dynamic test data and true labels
# x_test = np.load(config.DYNAMIC_IF_KMEANS_TRAIN_FILE)
# y_test = np.load(os.path.join(config.PROCESSED_DYNAMIC_DIR, 'y_test_dynamic.npy'))

# print("=== ⚡️ Isolation Forest Evaluation (dynamic) ===")
# isolation_forest = joblib.load(config.ISOLATION_FOREST_MODEL_PATH)
# # Isolation Forest predicts -1 for anomalies, convert to 1 for evaluation
# y_pred_if = (isolation_forest.predict(x_test) == -1).astype(int)

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_if))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_if))

# try:
#     auc_if = roc_auc_score(y_test, y_pred_if)
#     print(f"ROC AUC: {auc_if:.4f}")
# except ValueError:
#     print("ROC AUC: Cannot be computed (only one class present in y_test)")

# print("\n=== ⚡️ KMeans Evaluation (dynamic) ===")
# kmeans = joblib.load(config.KMEANS_MODEL_PATH)

# # Calculate distances between each point and its assigned cluster center
# distances = np.linalg.norm(x_test - kmeans.cluster_centers_[kmeans.predict(x_test)], axis=1)
# threshold = np.percentile(distances, 95)  # arbitrary threshold at 95th percentile
# # Points with distance above threshold are considered anomalies
# y_pred_kmeans = (distances > threshold).astype(int)

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_kmeans))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_kmeans))

# try:
#     auc_km = roc_auc_score(y_test, y_pred_kmeans)
#     print(f"ROC AUC: {auc_km:.4f}")
# except ValueError:
#     print("ROC AUC: Cannot be computed")
