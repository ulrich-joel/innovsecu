# src/main.py
import os
import pandas as pd
from config.config import Config
from .preprocessing.processor_static import clean_static_data
from .preprocessing.processor_dynamic import DynamicLogPreprocessor
from .mitre_mapper import run_mitre_mapping
from .clustering import ClusteringModel
from .logger import logger

import traceback

def load_and_concat_csv(path):
    import glob
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Path {path} not found.")

def feature_engineering(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return df[numeric_cols]

def main():
    config = Config()
    paths = config.DATASETS

    # === 1. Static Preprocessing ===
    if not os.path.exists(paths['static_cleaned']):
        logger.info("Starting static data preprocessing...")
        clean_static_data(paths['static_input'], paths['static_cleaned'])
        logger.info("Static data preprocessing completed.")
    else:
        logger.info("Static preprocessing already done. Skipping...")

    # === 2. Dynamic Preprocessing ===
    if not os.path.exists(paths['dynamic_llm_output']):
        logger.info("Starting dynamic data preprocessing...")
        processor = DynamicLogPreprocessor(paths['dynamic_input_dir'], paths['dynamic_processed_dir'])
        processor.preprocess()
        logger.info("Dynamic data preprocessing completed.")
    else:
        logger.info("Dynamic LLM dataset already exists. Skipping...")

    # === 3. Static Clustering ===
    try:
        logger.info("Loading cleaned static data for clustering...")
        static_df = load_and_concat_csv(paths['static_cleaned'])
        features_static = feature_engineering(static_df)

        logger.info(f"Features used: {features_static.columns.tolist()}")
        logger.info(f"Shape: {features_static.shape}")

        clustering_static = ClusteringModel(n_clusters=6)
        static_clustered = clustering_static.fit_predict(features_static)

        # Safety check: ensure the path is not a directory
        if os.path.isdir(paths['static_clustered']):
            import shutil
            shutil.rmtree(paths['static_clustered'])
            logger.warning(f"{paths['static_clustered']} was a directory. Removed it.")

        static_clustered.to_csv(paths['static_clustered'], index=False)
        logger.info(f"Static clustering completed and saved to {paths['static_clustered']}")
    except Exception as e:
        logger.error(f"Static clustering failed: {e}")
        logger.error(traceback.format_exc())

    # === 4. Dynamic Clustering ===
    try:
        logger.info("Loading dynamic intermediate data for clustering...")
        dynamic_df = load_and_concat_csv(paths['dynamic_intermediate_folder'])
        features_dynamic = feature_engineering(dynamic_df)

        clustering_dynamic = ClusteringModel(n_clusters=6)
        dynamic_clustered = clustering_dynamic.fit_predict(features_dynamic)
        dynamic_clustered.to_csv(paths['dynamic_clustered'], index=False)

        logger.info(f"Dynamic clustering completed and saved to {paths['dynamic_clustered']}")
    except Exception as e:
        logger.error(f"Dynamic clustering failed: {e}")
        logger.error(traceback.format_exc())

    # === 5. MITRE Mapping ===
    try:
        run_mitre_mapping(
            static_file=paths['static_clustered'],
            dynamic_file=paths['dynamic_clustered'],
            mapping_file=paths['mitre_mapping_file'],
            output_file=paths['mapped_output']
        )
        logger.info(f"MITRE mapping completed and saved to {paths['mapped_output']}")
    except Exception as e:
        logger.error(f"MITRE mapping failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
