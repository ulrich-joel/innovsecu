# src/preprocessing/preprocess_combined.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config.config import Config

config = Config()

STATIC_FILE = config.STATIC_FILE
DYNAMIC_FILE = config.DYNAMIC_FILE
OUTPUT_FILE = config.COMBINED_FEATURES_FILE

def preprocess():
    print("📦 Loading raw files...")
    df_static = pd.read_csv(STATIC_FILE)
    df_dynamic = pd.read_csv(DYNAMIC_FILE)

    print(f"✅ Static: {df_static.shape}, Dynamic: {df_dynamic.shape}")

    # Use processName <=> FileName (without extension) as the join key
    df_static['processName'] = df_static['FileName'].apply(lambda x: os.path.splitext(x)[0].lower())
    df_dynamic['processName'] = df_dynamic['processName'].str.lower()

    print("🔗 Merging on 'processName'...")
    df_merged = pd.merge(df_dynamic, df_static, on='processName', how='inner')
    print(f"✅ Merged shape: {df_merged.shape}")

    # Select useful columns (dynamic numerical + selected static features)
    dyn_features = ['argsNum', 'returnValue', 'sus', 'evil', 'behavior_cluster', 'stack_depth']
    stat_features = ['Machine','DebugSize','DebugRVA','MajorImageVersion','MajorOSVersion',
                     'ExportRVA','ExportSize','IatVRA','MajorLinkerVersion','MinorLinkerVersion',
                     'NumberOfSections','SizeOfStackReserve','DllCharacteristics','ResourceSize']

    final_features = dyn_features + stat_features
    data = df_merged[final_features].fillna(0)

    # ⚖️ Standardize (mean = 0, std = 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 💾 Save to file
    np.save(OUTPUT_FILE, scaled_data)
    print(f"✅ Combined data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess()
