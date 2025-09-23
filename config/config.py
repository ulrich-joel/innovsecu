import os, logging

class Config:
    def __init__(self):
        # Base directory
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # === 📁 Dossiers principaux ===
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.PROCESSED_STATIC_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'processed_static')  # corrigé le nom
        self.PROCESSED_DYNAMIC_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'processed_dynamic')
        self.LLM_READY_DIR = os.path.join(self.DATA_DIR, 'llm_ready')

        # === 📄 Fichiers de données brutes ===
        self.RAW_STATIC_DATA = os.path.join(self.RAW_DATA_DIR, 'data_file.csv')
        self.DYNAMIC_FILE = os.path.join(self.RAW_DATA_DIR, 'labelled_training_data.csv')

        # === 📄 Fichiers de données traitées ===
        self.X_TEST_FILE = os.path.join(self.PROCESSED_STATIC_DIR, 'x_test.npy')
        self.Y_TEST_FILE = os.path.join(self.PROCESSED_STATIC_DIR, 'y_test.npy')
        self.IF_KMEANS_TRAIN_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_train.npy')
        self.IF_KMEANS_VAL_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_val.npy')
        self.IF_KMEANS_TEST_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'if_kmeans_test.npy')
        self.DYNAMIC_IF_KMEANS_TRAIN_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'dynamic_if_kmeans_train.npy')
        self.COMBINED_FEATURES_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'combined_features.npy')

        # === 🧠 VAE ===
        self.VAE_FEATURES_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'vae_features.pt')
        self.VAE_VALIDATION_FILE = os.path.join(self.PROCESSED_DYNAMIC_DIR, 'vae_validation.pt')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'output', 'models')
        self.BEST_VAE_MODEL_PATH = os.path.join(self.MODELS_DIR, 'best_vae_model.pth')


        # === 📦 Modèles ===
        self.ISOLATION_FOREST_MODEL_PATH = os.path.join(self.MODELS_DIR, 'isolation_forest.pkl')
        self.KMEANS_MODEL_PATH = os.path.join(self.MODELS_DIR, 'kmeans.pkl')
        self.BEST_ISOFOREST_MODEL_PATH = self.ISOLATION_FOREST_MODEL_PATH
        self.BEST_KMEANS_MODEL_PATH = self.KMEANS_MODEL_PATH


        # === 📊 Reporting & logs ===
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'output')
        self.REPORTS_DIR = os.path.join(self.OUTPUT_DIR, 'reports')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')

        # === 🗺 Mapping MITRE ===
        self.MAPPING_DIR = os.path.join(self.BASE_DIR, 'mapping')
        self.MITRE_MAPPING_FILE = os.path.join(self.MAPPING_DIR, 'mapping_mitre.json')

        # === 📑 Fichiers groupés ===
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

        # === 🔧 Paramètres techniques ===
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
        """Crée tous les dossiers nécessaires si absents."""
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
            self.MAPPING_DIR
        ]:
            os.makedirs(path, exist_ok=True)

class ModelTrainer:
    def __init__(self, device="cpu", config=None):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model = None
        self.isolation_forest = None
        self.kmeans = None
        self.config = config or Config()  # ✅ FIXED
