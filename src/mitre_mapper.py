import pandas as pd
import json
from src.logger import logger  # ← Centralized logging system

class MitreMapper:
    def __init__(self, mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
                # Example structure:
                # {
                #   "eventId_to_mitre": {...},
                #   "cluster_to_mitre": {...}
                # }
            logger.info(f"Successfully loaded mapping file: {mapping_file}")
        except FileNotFoundError:
            logger.error(f"Mapping file '{mapping_file}' not found.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Mapping file '{mapping_file}' is not valid JSON.")
            raise

    def map_events(self, df_dynamic):
        def get_techniques(eventId):
            eventId_str = str(eventId)
            return self.mapping.get("eventId_to_mitre", {}).get(eventId_str, [])

        df_dynamic['MITRE_Techniques'] = df_dynamic['eventId'].apply(get_techniques)
        logger.info("Mapped eventId to MITRE techniques.")
        return df_dynamic

    def map_clusters(self, df_dynamic):
        def get_techniques(cluster):
            cluster_str = str(cluster)
            return self.mapping.get("cluster_to_mitre", {}).get(cluster_str, [])

        df_dynamic['MITRE_Techniques_Cluster'] = df_dynamic['Cluster'].apply(get_techniques)
        logger.info("Mapped clusters to MITRE techniques.")
        return df_dynamic

def run_mitre_mapping(static_file, dynamic_file, mapping_file, output_file):
    try:
        static_df = pd.read_csv(static_file)
        dynamic_df = pd.read_csv(dynamic_file)
        logger.info("Successfully loaded static and dynamic datasets.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return
    except pd.errors.EmptyDataError:
        logger.error("One of the input files is empty.")
        return
    except Exception as e:
        logger.error(f"Error reading input files: {e}")
        return

    # Check if the required columns exist
    missing_columns = []
    for col in ['eventId', 'Cluster']:
        if col not in dynamic_df.columns:
            missing_columns.append(f"{col} in dynamic_df")
    for col in ['Cluster', 'Benign']:
        if col not in static_df.columns:
            missing_columns.append(f"{col} in static_df")

    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    try:
        # Load mapping
        mapper = MitreMapper(mapping_file)

        # Apply MITRE technique mappings
        dynamic_df = mapper.map_events(dynamic_df)
        dynamic_df = mapper.map_clusters(dynamic_df)

        # Remove duplicates in static_df to prevent memory explosion
        static_unique = static_df[['Cluster', 'Benign']].drop_duplicates(subset=['Cluster'])

        # Estimate the size of the merge to avoid memory issues
        estimated_rows = len(dynamic_df) * static_unique['Cluster'].nunique()
        if estimated_rows > 1e8:  # 100 million rows
            logger.warning(f"MITRE mapping aborted: estimated merge size too large ({estimated_rows} rows).")
            return

        # Safe merge
        merged_df = dynamic_df.merge(static_unique, on='Cluster', how='left')

        # Convert technique lists to semicolon-separated strings
        for col in ['MITRE_Techniques', 'MITRE_Techniques_Cluster']:
            merged_df[col] = merged_df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else '')

        # Export to CSV
        merged_df.to_csv(output_file, index=False)
        logger.info(f"MITRE mapping completed and saved to {output_file}")

    except Exception as e:
        logger.error(f"Unexpected error during MITRE mapping: {e}")
