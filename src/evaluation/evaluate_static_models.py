# src/evaluation/evaluate_static_models.py
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from config.config import Config
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os

# Load configuration settings
config = Config()

# Load test data and convert to binary labels (0 for benign, 1 for malicious)
X_test = np.load(config.X_TEST_FILE)
y_test = (np.load(config.Y_TEST_FILE) > 0).astype(int)

print(f"📊 Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

def load_or_retrain_model(model_path, model_class, X_train, **kwargs):
    """
    Load a pre-trained model or retrain it if not available or incompatible.
    
    Args:
        model_path (str): Path to the model file
        model_class: Model class (IsolationForest or KMeans)
        X_train (array): Training data for retraining if needed
        **kwargs: Model parameters
    
    Returns:
        Trained model instance
    """
    try:
        # Try to load existing model
        model = joblib.load(model_path)
        # Check if model dimensions match current data
        if hasattr(model, 'n_features_in_') and model.n_features_in_ == X_train.shape[1]:
            print(f"✅ Model loaded: {os.path.basename(model_path)} ({model.n_features_in_} features)")
            return model
        else:
            print(f"🔄 Incompatible dimensions, retraining...")
    except Exception as e:
        print(f"🔄 Model not found or corrupted ({e}), creating new model...")
    
    # Retrain model if loading failed or dimensions don't match
    model = model_class(**kwargs)
    model.fit(X_train)
    
    # Create directory if it doesn't exist and save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model retrained and saved: {os.path.basename(model_path)} ({model.n_features_in_} features)")
    return model

# Load training data for potential retraining
try:
    X_train = np.load(config.X_TRAIN_FILE)
    print(f"📊 Training data: {X_train.shape}")
except:
    print("⚠️ Training data not available, using test data")
    X_train = X_test

# Load or retrain models with different contamination levels
print("\n🎯 Loading models...")

# Dictionary to store models with different contamination levels
models = {}

# Train multiple Isolation Forest models with different contamination levels
contamination_levels = [0.01, 0.05, 0.10, 0.15]  # 1%, 5%, 10%, 15%

for cont in contamination_levels:
    model_name = f"IF_cont={int(cont*100)}"
    models[model_name] = load_or_retrain_model(
        f"{config.ISOLATION_FOREST_MODEL_PATH}_{int(cont*100)}",
        IsolationForest,
        X_train,
        contamination=cont,  # Different contamination levels
        random_state=42
    )

# Load KMeans model
kmeans = load_or_retrain_model(
    config.KMEANS_MODEL_PATH, 
    KMeans,
    X_train,
    n_clusters=2,
    random_state=42
)

# Store results for comparison
results_summary = []

# === EVALUATE ISOLATION FOREST MODELS WITH DIFFERENT CONTAMINATION LEVELS ===
print("\n" + "="*50)
print("🔍 Evaluating Isolation Forest models:")
print("="*50)

for model_name, model in models.items():
    print(f"\n== {model_name} ==")
    
    # Get predictions (Isolation Forest returns -1 for anomalies, 1 for normal)
    if_raw_preds = model.predict(X_test)
    if_preds = np.where(if_raw_preds == -1, 1, 0)  # Convert to 1 for anomalies, 0 for normal
    
    # Calculate metrics
    precision = precision_score(y_test, if_preds, zero_division=0)
    recall = recall_score(y_test, if_preds, zero_division=0)
    f1 = f1_score(y_test, if_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, if_preds)
    except ValueError:
        auc = 0.5  # Default value for undefined AUC
    
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    
    # Store results for summary
    results_summary.append({
        'name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'detections': np.sum(if_preds)
    })

# === EVALUATE KMEANS MODEL ===
print("\n" + "="*50)
print("🔍 Evaluating KMeans model:")
print("="*50)

# KMeans anomaly detection based on distance to cluster centers
kmeans_labels = kmeans.predict(X_test)
distances = np.linalg.norm(X_test - kmeans.cluster_centers_[kmeans_labels], axis=1)
threshold = np.percentile(distances, 95)  # 95th percentile threshold
kmeans_preds = (distances > threshold).astype(int)

# Calculate KMeans metrics
precision = precision_score(y_test, kmeans_preds, zero_division=0)
recall = recall_score(y_test, kmeans_preds, zero_division=0)
f1 = f1_score(y_test, kmeans_preds, zero_division=0)

try:
    auc = roc_auc_score(y_test, kmeans_preds)
except ValueError:
    auc = 0.5

print(f"== KMeans (95th percentile) ==")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

# Store KMeans results
results_summary.append({
    'name': 'KMeans (95th percentile)',
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'detections': np.sum(kmeans_preds)
})

# === PERFORMANCE SUMMARY COMPARISON ===
print("\n" + "="*50)
print("📊 Model Performance Summary:")
print("="*50)

print(f"{'name':<25} {'precision':<10} {'recall':<10} {'f1':<10} {'auc':<10}")
print("-" * 65)
for result in results_summary:
    print(f"{result['name']:<25} {result['precision']:<10.3f} {result['recall']:<10.3f} {result['f1']:<10.3f} {result['auc']:<10.3f}")

print("\n✅ Static model evaluation completed!")

# Use the best performing Isolation Forest model for visualizations
best_if_model = models['IF_cont=15']  # Using 15% contamination for visualization
if_raw_preds = best_if_model.predict(X_test)
if_preds = np.where(if_raw_preds == -1, 1, 0)

# === VISUALIZATIONS - ROC AND PR CURVES ===
print("\n" + "="*50)
print("=== GENERATING ROC AND PR CURVES ===")
print("="*50)

try:
    # Create 2x2 subplot for curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Isolation Forest - ROC Curve
    RocCurveDisplay.from_predictions(y_test, if_preds, ax=axes[0, 0])
    axes[0, 0].set_title("ROC Curve - Isolation Forest\n(AUC = {:.3f})".format(
        roc_auc_score(y_test, if_preds) if len(np.unique(y_test)) > 1 else 0.5
    ))
    axes[0, 0].grid(True)
    
    # Isolation Forest - Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, if_preds, ax=axes[0, 1])
    axes[0, 1].set_title("Precision-Recall Curve - Isolation Forest")
    axes[0, 1].grid(True)
    
    # KMeans - ROC Curve
    RocCurveDisplay.from_predictions(y_test, distances, ax=axes[1, 0])
    axes[1, 0].set_title("ROC Curve - KMeans\n(AUC = {:.3f})".format(
        roc_auc_score(y_test, distances) if len(np.unique(y_test)) > 1 else 0.5
    ))
    axes[1, 0].grid(True)
    
    # KMeans - Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, distances, ax=axes[1, 1])
    axes[1, 1].set_title("Precision-Recall Curve - KMeans")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('static_model_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ ROC and Precision-Recall curves saved (static_model_curves.png)")
    
except Exception as e:
    print(f"⚠️ Error generating curves: {e}")

# === VISUALIZATIONS - CONFUSION MATRICES ===
print("\n" + "="*50)
print("=== GENERATING CONFUSION MATRICES ===")
print("="*50)

try:
    # Create separate figure for confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Isolation Forest - Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, if_preds, ax=ax1, cmap='Blues', 
                                          values_format='d', colorbar=False)
    ax1.set_title("Confusion Matrix - Isolation Forest", fontsize=14, fontweight='bold')
    
    # KMeans - Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, kmeans_preds, ax=ax2, cmap='Blues', 
                                          values_format='d', colorbar=False)
    ax2.set_title("Confusion Matrix - KMeans", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrices saved (static_confusion_matrices.png)")
    
except Exception as e:
    print(f"⚠️ Error generating confusion matrices: {e}")

print("\n🎉 Complete static model evaluation finished successfully!")