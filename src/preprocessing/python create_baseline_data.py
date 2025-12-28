import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config.config import Config

def create_baseline_data():
    """
    Crée la Version A : Données brutes sans mapping MITRE
    """
    print("🚀 Création des données baseline (Version A)...")
    
    # Initialisation config
    config = Config()
    config.ensure_dirs()
    
    # Charger les données dynamiques brutes
    print("📥 Chargement des données dynamiques brutes...")
    df_baseline = pd.read_csv(config.DYNAMIC_FILE)
    
    # Vérifier les colonnes disponibles
    print("🔍 Colonnes disponibles:", df_baseline.columns.tolist())
    
    # Définir les features de base (identique à votre train_dynamic_models.py)
    features_baseline = ['processId', 'threadId', 'parentProcessId', 'userId', 'eventId', 
                        'argsNum', 'returnValue', 'behavior_cluster', 'stack_depth']
    
    # Vérifier que toutes les features existent
    missing_features = [f for f in features_baseline if f not in df_baseline.columns]
    if missing_features:
        raise ValueError(f"❌ Features manquantes: {missing_features}")
    
    # Colonne cible
    if 'sus' not in df_baseline.columns:
        raise ValueError("❌ Colonne 'sus' manquante")
    
    # Extraire features et target
    X_baseline = df_baseline[features_baseline]
    y_baseline = df_baseline['sus']
    
    print(f"✅ Données brutes chargées: {X_baseline.shape[0]} échantillons, {X_baseline.shape[1]} features")
    
    # Normalisation (identique à votre code existant)
    print("🔄 Normalisation des données...")
    scaler = StandardScaler()
    X_baseline_scaled = scaler.fit_transform(X_baseline)
    
    # Split train/test (80/20) - même méthode que votre code
    split_idx = int(0.8 * len(X_baseline_scaled))
    
    X_train_baseline = X_baseline_scaled[:split_idx]
    X_test_baseline = X_baseline_scaled[split_idx:]
    y_train_baseline = y_baseline[:split_idx]
    y_test_baseline = y_baseline[split_idx:]
    
    print(f"📊 Split train/test: {len(X_train_baseline)} train, {len(X_test_baseline)} test")
    
    # Sauvegarder les données baseline
    output_dir = os.path.join(config.PROCESSED_DATA_DIR, "comparative_experiment")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder en numpy
    np.save(os.path.join(output_dir, "X_train_baseline.npy"), X_train_baseline)
    np.save(os.path.join(output_dir, "X_test_baseline.npy"), X_test_baseline)
    np.save(os.path.join(output_dir, "y_train_baseline.npy"), y_train_baseline)
    np.save(os.path.join(output_dir, "y_test_baseline.npy"), y_test_baseline)
    
    # Sauvegarder le scaler
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, "baseline_scaler.pkl"))
    
    # Sauvegarder les noms de features
    with open(os.path.join(output_dir, "baseline_features.txt"), "w") as f:
        f.write("\n".join(features_baseline))
    
    # Statistiques descriptives
    print("\n📈 Statistiques des données baseline:")
    print(f"   - Shape X_train: {X_train_baseline.shape}")
    print(f"   - Shape X_test: {X_test_baseline.shape}")
    print(f"   - Distribution y_train: {np.bincount(y_train_baseline)}")
    print(f"   - Distribution y_test: {np.bincount(y_test_baseline)}")
    print(f"   - Features utilisées: {features_baseline}")
    
    print(f"\n💾 Données baseline sauvegardées dans: {output_dir}")
    
    return {
        'X_train': X_train_baseline,
        'X_test': X_test_baseline,
        'y_train': y_train_baseline,
        'y_test': y_test_baseline,
        'features': features_baseline,
        'scaler': scaler
    }

def verify_baseline_data():
    """
    Vérifie que les données baseline ont été créées correctement
    """
    config = Config()
    output_dir = os.path.join(config.PROCESSED_DATA_DIR, "comparative_experiment")
    
    print("\n🔍 Vérification des données baseline...")
    
    try:
        X_train = np.load(os.path.join(output_dir, "X_train_baseline.npy"))
        X_test = np.load(os.path.join(output_dir, "X_test_baseline.npy"))
        y_train = np.load(os.path.join(output_dir, "y_train_baseline.npy"))
        y_test = np.load(os.path.join(output_dir, "y_test_baseline.npy"))
        
        print(f"✅ X_train: {X_train.shape}")
        print(f"✅ X_test: {X_test.shape}")
        print(f"✅ y_train: {y_train.shape} (distribution: {np.bincount(y_train)})")
        print(f"✅ y_test: {y_test.shape} (distribution: {np.bincount(y_test)})")
        
        # Vérifier la normalisation
        print(f"✅ Mean X_train: {np.mean(X_train, axis=0)[:3]}... (devrait être ~0)")
        print(f"✅ Std X_train: {np.std(X_train, axis=0)[:3]}... (devrait être ~1)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False

if __name__ == "__main__":
    # Créer les données baseline
    baseline_data = create_baseline_data()
    
    # Vérifier
    verify_baseline_data()
    
    print("\n🎯 Version A (Baseline) prête! Vous pouvez maintenant créer la Version B avec MITRE.")