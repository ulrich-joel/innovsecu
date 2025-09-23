import matplotlib.pyplot as plt
import numpy as np

# Remplacez ces valeurs par celles extraites de vos graphiques de feature importance
# Exemple simulé :
feature_names = ["IF_1%", "IF_5%", "IF_10%", "IF_15%", "KMeans"]
logistic_importances = np.array([0.4, 0.4, 0.4, 0.4, 0.5])
rf_importances = np.array([0.14, 0.16, 0.16, 0.15, 0.40])
svm_importances = np.array([0.028, 0.028, 0.028, 0.028, 0.012])

# Tracer les graphiques comparatifs
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
ax[0].bar(feature_names, logistic_importances, color="skyblue")
ax[0].set_title("Logistic Regression")
ax[0].set_ylabel("Importance")

ax[1].bar(feature_names, rf_importances, color="lightgreen")
ax[1].set_title("Random Forest")

ax[2].bar(feature_names, svm_importances, color="lightcoral")
ax[2].set_title("SVM (Permutation Importance)")

for a in ax:
    a.set_xticklabels(feature_names, rotation=45)

plt.tight_layout()
plt.show()
