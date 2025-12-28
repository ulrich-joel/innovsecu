# src/clustering.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .logger import logger
import pandas as pd


class ClusteringModel:
    """
    A class to implement clustering using the KMeans algorithm.

    This class standardizes the input data and applies the KMeans algorithm
    to group the data into clusters.

    Attributes:
        n_clusters (int): The number of clusters to create.
        scaler (StandardScaler): A scaler to normalize the data.
        model (KMeans): The KMeans model used for clustering.
    """

    def __init__(self, n_clusters=5):
        """
        Initializes the ClusteringModel with the specified number of clusters.

        Args:
            n_clusters (int, optional): The number of clusters to create. Defaults to 5.
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def fit_predict(self, dataset):
        """
        Applies clustering to the given dataset.

        This method selects numerical columns, standardizes the data,
        applies the KMeans algorithm, and returns a DataFrame with the assigned clusters.

        Args:
            dataset (pd.DataFrame): The input dataset as a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with an additional 'Cluster' column indicating
                          the cluster assigned to each row.

        Raises:
            Exception: If an error occurs during clustering.
        """
        try:
            # Select numerical columns
            numeric_columns = dataset.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numerical columns found in the dataset.")

            # Standardize the data
            numeric_data = dataset[numeric_columns]
            scaled_data = self.scaler.fit_transform(numeric_data)

            # Apply KMeans clustering
            clusters = self.model.fit_predict(scaled_data)

            # Add the cluster assignments to the DataFrame
            result_df = dataset.copy()
            result_df['Cluster'] = clusters

            logger.info(f"Clustering completed. {self.n_clusters} clusters created.")
            return result_df

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

