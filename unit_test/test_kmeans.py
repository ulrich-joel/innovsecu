# test_kmeans.py
import unittest
import numpy as np
from sklearn.cluster import KMeans
from config.config import Config

class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Charger la configuration
        self.cfg = Config()

        # Charger des données de test (par exemple, statiques traitées)
        self.X_train = np.random.rand(100, 15)  # Remplacer par des données réelles

        # Initialisation du modèle
        self.kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300)
        self.kmeans.fit(self.X_train)

    def test_clusters(self):
        # Vérifier qu'il y a bien 10 clusters
        self.assertEqual(len(set(self.kmeans.labels_)), 10, "Should have 10 clusters")

    def test_new_prediction(self):
        # Créer des données nouvelles à tester
        X_new = np.random.rand(10, 15)  # Remplacer par des données réelles

        # Prédire les clusters des nouvelles données
        predictions = self.kmeans.predict(X_new)

        # Vérifier que la prédiction des clusters fonctionne
        self.assertEqual(len(predictions), 10, "There should be 10 predictions for 10 data points")
    
if __name__ == '__main__':
    unittest.main()
