import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample
import os
from config.config import Config
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from src.evaluation.fusion import fuse_predictions 
import torch
import logging

logger = logging.getLogger("fusion")
logging.basicConfig(level=logging.INFO)

def load_model_outputs(trainer, X_test):
    # VAE
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
    with torch.no_grad():
        recon_x, _, _ = trainer.model(X_tensor)
    recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()

    # IF
    if_scores = -trainer.isolation_forest.decision_function(X_test)

    # KMeans
    distances = trainer.kmeans.transform(X_test).min(axis=1)

    return recon_errors, if_scores, distances

def get_predictions(recon_errors, if_scores, distances, thresholds):
    vae_preds = (recon_errors > thresholds["vae"]).astype(int)
    if_preds = (if_scores > thresholds["if"]).astype(int)
    km_preds = (distances > thresholds["kmeans"]).astype(int)
    return vae_preds, if_preds, km_preds

def combine_predictions(vae, ifo, kmeans, mode="or"):
    if mode == "or":
        return (vae | ifo | kmeans).astype(int)
    elif mode == "and":
        return (vae & ifo & kmeans).astype(int)
    elif mode == "majority":
        return ((vae + ifo + kmeans) >= 2).astype(int)
    else:
        raise ValueError("Mode invalide. Utilise 'or', 'and' ou 'majority'.")

def balance_test_data(X_test, y_test):
    X_benign = X_test[y_test == 0]
    X_malicious = X_test[y_test == 1]
    X_malicious_down = resample(X_malicious, replace=False, n_samples=len(X_benign), random_state=42)
    y_benign = np.zeros(len(X_benign))
    y_malicious_down = np.ones(len(X_malicious_down))
    X_balanced = np.concatenate([X_benign, X_malicious_down])
    y_balanced = np.concatenate([y_benign, y_malicious_down])
    return X_balanced, y_balanced

def fusion_evaluation(trainer, X_test, y_test, balanced=False):
    print("⚖️ Rééquilibrage du jeu de test avant fusion (downsampling)..." if balanced else "➡️ Évaluation sans rééquilibrage")

    if balanced:
        idx_0 = np.where(y_test == 0)[0]
        idx_1 = np.where(y_test == 1)[0]
        n = min(len(idx_0), len(idx_1))
        selected_idx = np.concatenate([
            np.random.choice(idx_0, n, replace=False),
            np.random.choice(idx_1, n, replace=False)
        ])
        np.random.shuffle(selected_idx)
        X_test = X_test[selected_idx]
        y_test = y_test[selected_idx]
        print(f"🧪 Nouveau jeu équilibré : {np.bincount(y_test)}")

    fusion_results = fuse_predictions(trainer, X_test, y_test, mode="all")

    summary = []
    print("🔁 Fusion des prédictions...")

    for mode, (preds, auc) in fusion_results.items():
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"\n📊 Mode: {mode.upper()}")
        print(f"AUC: {auc:.4f}")
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print("📄 Rapport de classification :")
        print(classification_report(y_test, preds))

        summary.append([mode.upper(), auc, precision, recall, f1])

    df_summary = pd.DataFrame(summary, columns=["Fusion Mode", "AUC", "Precision", "Recall", "F1"])

    timestamp = getattr(trainer, "timestamp", "latest")
    config = Config()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    output_path = os.path.join(
        config.REPORTS_DIR,
        "fusion_metrics_balanced.csv" if balanced else "fusion_metrics_full.csv"
    )

    df_summary.to_csv(output_path, index=False)
    print(f"\n✅ Résumé enregistré dans {output_path}")





# def fusion_evaluation(trainer, X_test, y_test, balanced=False):
#     if balanced:
#         logger.info("⚖️ Rééquilibrage du jeu de test avant fusion (downsampling)...")
#         X_test, y_test = balance_test_data(X_test, y_test)
#         print(f"🧪 Nouveau jeu équilibré : {np.bincount(y_test.astype(int))}")

#     logger.info("🔀 Fusion des prédictions...")
#     recon_errors, if_scores, distances = load_model_outputs(trainer, X_test)

#     thresholds = {
#         "vae": np.percentile(recon_errors, 95),
#         "if": np.percentile(if_scores, 95),
#         "kmeans": np.percentile(distances, 95)
#     }

#     vae_pred, if_pred, km_pred = get_predictions(recon_errors, if_scores, distances, thresholds)

#     fusion_results = []

#     for mode in ["or", "majority", "and"]:
#         fused = combine_predictions(vae_pred, if_pred, km_pred, mode=mode)
#         auc = roc_auc_score(y_test, (vae_pred + if_pred + km_pred) / 3)
#         report = classification_report(y_test, fused, output_dict=True)

#         fusion_results.append({
#             "Fusion Mode": mode.upper(),
#             "Accuracy": report["accuracy"],
#             "Precision_0": report["0"]["precision"],
#             "Recall_0": report["0"]["recall"],
#             "F1_0": report["0"]["f1-score"],
#             "Precision_1": report["1"]["precision"],
#             "Recall_1": report["1"]["recall"],
#             "F1_1": report["1"]["f1-score"],
#             "AUC": auc
#         })

#         logger.info(f"\n📌 Mode de fusion : {mode.upper()}")
#         logger.info("\n" + classification_report(y_test, fused, digits=4))
#         logger.info(f"AUC approx (score moyen des 3) : {auc:.4f}")

#         print(f"\n📌 Mode de fusion : {mode.upper()}")
#         print(classification_report(y_test, fused, digits=4))
#         print(f"AUC approx (score moyen des 3) : {auc:.4f}")

#     df_results = pd.DataFrame(fusion_results)
#     suffix = "balanced" if balanced else "full"
#     output_path = os.path.join(Config().REPORTS_DIR, f"fusion_metrics_{suffix}.csv")
#     df_results.to_csv(output_path, index=False)
#     print(f"📁 Résultats de fusion sauvegardés dans : {output_path}")
