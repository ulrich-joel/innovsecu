# src/evaluation/fusion.py

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def load_model_outputs(trainer, X_test):
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
    with torch.no_grad():
        recon_x, _, _ = trainer.model(X_tensor)
    recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()
    if_scores = -trainer.isolation_forest.decision_function(X_test)
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

def fuse_predictions(trainer, X_test, y_test=None, mode="all"):
    thresholds = trainer.thresholds
    recon_errors, if_scores, distances = load_model_outputs(trainer, X_test)
    vae_preds, if_preds, km_preds = get_predictions(recon_errors, if_scores, distances, thresholds)

    fusion_modes = {
        "or": combine_predictions(vae_preds, if_preds, km_preds, mode="or"),
        "and": combine_predictions(vae_preds, if_preds, km_preds, mode="and"),
        "majority": combine_predictions(vae_preds, if_preds, km_preds, mode="majority"),
    }

    results = {}
    for key, preds in fusion_modes.items():
        auc = roc_auc_score(y_test, preds) if y_test is not None else None
        results[key] = (preds, auc)
    return results



# import numpy as np
# import torch
# from sklearn.metrics import roc_auc_score

# def load_model_outputs(trainer, X_test):
#     X_tensor = torch.tensor(X_test, dtype=torch.float32).to(trainer.device)
#     with torch.no_grad():
#         recon_x, _, _ = trainer.model(X_tensor)
#     recon_errors = ((X_tensor - recon_x) ** 2).mean(dim=1).cpu().numpy()
#     if_scores = -trainer.isolation_forest.decision_function(X_test)
#     distances = trainer.kmeans.transform(X_test).min(axis=1)
#     return recon_errors, if_scores, distances

# def get_predictions(recon_errors, if_scores, distances, thresholds):
#     vae_preds = (recon_errors > thresholds["vae"]).astype(int)
#     if_preds = (if_scores > thresholds["if"]).astype(int)
#     km_preds = (distances > thresholds["kmeans"]).astype(int)
#     return vae_preds, if_preds, km_preds

# def combine_predictions(vae, ifo, kmeans, mode="or"):
#     if mode == "or":
#         return (vae | ifo | kmeans).astype(int)
#     elif mode == "and":
#         return (vae & ifo & kmeans).astype(int)
#     elif mode == "majority":
#         return ((vae + ifo + kmeans) >= 2).astype(int)
#     else:
#         raise ValueError("Mode invalide. Utilise 'or', 'and' ou 'majority'.")

# def fuse_predictions(trainer, X_test, y_test=None, mode="all"):
#     thresholds = trainer.thresholds
#     recon_errors, if_scores, distances = load_model_outputs(trainer, X_test)
#     vae_preds, if_preds, km_preds = get_predictions(recon_errors, if_scores, distances, thresholds)

#     fusion_modes = {
#         "or": combine_predictions(vae_preds, if_preds, km_preds, mode="or"),
#         "and": combine_predictions(vae_preds, if_preds, km_preds, mode="and"),
#         "majority": combine_predictions(vae_preds, if_preds, km_preds, mode="majority"),
#     }

#     results = {}
#     for key, preds in fusion_modes.items():
#         auc = roc_auc_score(y_test, preds) if y_test is not None else None
#         results[key] = (preds, auc)
#     return results

