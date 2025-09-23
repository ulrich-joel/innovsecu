# src/clustering.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .logger import logger
import pandas as pd


class ClusteringModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def fit_predict(self, dataset):
        """Fit the model and predict clusters."""
        try:
            numeric_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the dataset.")
            
            numeric_data = dataset[numeric_columns]
            scaled_data = self.scaler.fit_transform(numeric_data)
            clusters = self.model.fit_predict(scaled_data)
            
            result_df = dataset.copy()
            result_df['Cluster'] = clusters
            
            logger.info(f"Clustering completed. {self.n_clusters} clusters created.")
            return result_df
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

