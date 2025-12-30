# test_isolation_forest.py
import unittest
import numpy as np
from sklearn.ensemble import IsolationForest
from config.config import Config

class TestIsolationForest(unittest.TestCase):
    
    def setUp(self):
        # Charger la configuration
        self.cfg = Config()

        # Charger des données de test (par exemple, statiques traitées)
        self.X_train = np.random.rand(100, 15)  # Remplacer par des données réelles

        # Initialisation du modèle
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.1)
        self.iso_forest.fit(self.X_train)

    def test_anomaly_detection(self):
        # Créer des données nouvelles à tester
        X_test = np.random.rand(10, 15)  # Remplacer par des données réelles

        # Prédire les anomalies
        predictions = self.iso_forest.predict(X_test)

        # Vérifier qu'on a bien des résultats -1 (anomalie) ou 1 (normal)
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])), "Predictions should be -1 or 1")
    
if __name__ == '__main__':
    unittest.main()
