"""
config.py

Configuration class for paths, parameters, and model settings used across the project.

This class defines directory paths for raw and processed data, model output locations,
logging directories, and various hyperparameters for models such as VAE and clustering.

Author: Ngueyep Ulrich
Date: 2025-10-13
"""

import os
import logging

class Config:
    def __init__(self):
        # === Base directory ===
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # === Outputs (défini tôt car utilisé ensuite) ===
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'outputs')
        self.PROCESSED_DATA_DIR = os.path.join(self.OUTPUT_DIR, 'processed')
        self.PROCESSED_STATIC_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'processed_static')
        self.PROCESSED_DYNAMIC_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'processed_dynamic')
        self.EVALUATION_DIR = os.path.join(self.OUTPUT_DIR, 'evaluation')
        self.REPORTS_DIR = os.path.join(self.OUTPUT_DIR, 'reports')
        self.MODELS_DIR = os.path.join(self.OUTPUT_DIR, 'models')
        self.MODELS_DIR_STATIC_TRAIN = os.path.join(self.MODELS_DIR, 'train_static_model')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')

        # === Data directories ===
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.LLM_READY_DIR = os.path.join(self.DATA_DIR, 'llm_ready')

        # === Raw data files ===
        self.RAW_STATIC_DATA = os.path.join(self.RAW_DATA_DIR, 'data_file.csv')
        self.DYNAMIC_FILE = os.path.join(self.RAW_DATA_DIR, 'labelled_training_data.csv')

        # === Processed static ===
        self.X_TEST_FILE = os.path.join(self.PROCESSED_STATIC_DIR, 'x_test.npy')
        self.Y_TEST_FILE = os.path.join(self.PROCESSED_STATIC_DIR, 'y_test.npy')

        # === Processed dynamic ===
        self.IF_KMEANS_TRAIN_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_train.npy')
        self.IF_KMEANS_VAL_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_val.npy')
        self.IF_KMEANS_TEST_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_test.npy')
        self.DYNAMIC_IF_KMEANS_TRAIN_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'dynamic_if_kmeans_train.npy')
        self.COMBINED_FEATURES_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'combined_features.npy')

        # === VAE ===
        self.VAE_FEATURES_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'vae_features.pt')
        self.VAE_VALIDATION_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'vae_validation.pt')
        self.BEST_VAE_MODEL_PATH = os.path.join(self.MODELS_DIR, 'best_vae_model.pth')

        # === Models ===
        # Pour chercher les modèles dans le dossier 'train_static_model'
        self.ISOLATION_FOREST_MODEL_PATH = os.path.join(self.MODELS_DIR_STATIC_TRAIN, 'isolation_forest')
        self.KMEANS_MODEL_PATH = os.path.join(self.MODELS_DIR_STATIC_TRAIN, 'kmeans.pkl')
        self.BEST_ISOFOREST_MODEL_PATH = self.ISOLATION_FOREST_MODEL_PATH
        self.BEST_KMEANS_MODEL_PATH = self.KMEANS_MODEL_PATH

        # === MITRE Mapping ===
        self.MAPPING_DIR = os.path.join(self.BASE_DIR, 'mapping')
        self.MITRE_MAPPING_FILE = os.path.join(self.MAPPING_DIR, 'mapping_mitre.json')

        # === Grouped file references ===
        self.DATASETS = {
            'static_input': self.RAW_STATIC_DATA,
            'static_cleaned': os.path.join(self.PROCESSED_STATIC_DIR, 'static_cleaned.csv'),
            'static_clustered': os.path.join(self.MODELS_DIR, 'static_clustered.csv'),
            'dynamic_input_dir': self.RAW_DATA_DIR,
            'dynamic_processed_dir': self.PROCESSED_DYNAMIC_DIR,
            'dynamic_intermediate_folder': os.path.join(self.PROCESSED_DYNAMIC_DIR, 'intermediate'),
            'dynamic_cleaned_flag': os.path.join(self.PROCESSED_DYNAMIC_DIR, 'dynamic_cleaned_flag.txt'),
            'dynamic_llm_output': os.path.join(self.PROCESSED_DYNAMIC_DIR, 'dynamic_for_llm.jsonl'),
            'dynamic_clustered': os.path.join(self.MODELS_DIR, 'dynamic_clustered.csv'),
            'mapped_output': os.path.join(self.OUTPUT_DIR, 'dynamic_mitre_mapped.csv'),
            'mitre_mapping_file': self.MITRE_MAPPING_FILE,
        }

        # === Technical parameters ===
        self.CLUSTERING_PARAMS = {
            "n_clusters": 10,
            "init": "k-means++",
            "max_iter": 300,
        }

        self.VAE_CONFIG = {
            "input_dim": 15,
            "latent_dim": 8,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "beta_max": 0.5,
            "patience": 10
        }

        self.LOG_LEVEL = "INFO"
        self.TRAIN_RATIO = 0.7
        self.VALID_RATIO = 0.2
        self.TEST_RATIO = 0.1

    def ensure_dirs(self):
        """Create all necessary directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.PROCESSED_STATIC_DIR,
            self.PROCESSED_DYNAMIC_DIR,
            self.OUTPUT_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
            self.LOG_DIR,
            self.MAPPING_DIR,
            self.EVALUATION_DIR 
        ]:
            os.makedirs(path, exist_ok=True)

    def to_dict(self):
        return {
            "BASE_DIR": self.BASE_DIR,
            "OUTPUT_DIR": self.OUTPUT_DIR,
            "TRAIN_RATIO": self.TRAIN_RATIO,
            "VALID_RATIO": self.VALID_RATIO,
            "TEST_RATIO": self.TEST_RATIO,
            "VAE_CONFIG": self.VAE_CONFIG,
            "CLUSTERING_PARAMS": self.CLUSTERING_PARAMS,
            "LOG_LEVEL": self.LOG_LEVEL
        }

class ModelTrainer:
    def __init__(self, device="cpu", config=None):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.isolation_forest = None
        self.kmeans = None
        self.config = config or Config()  # Default fallback to Config
