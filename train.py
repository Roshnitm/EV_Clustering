import os
import joblib
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

from utils.feature_extractor import extract_features

train_path = "Dataset/Training_set/Training_set"

print("Extracting features for clustering...")
X_train, _ = extract_features(train_path)

# Define Clustering Models
models = {
    "KMeans": KMeans(n_clusters=2, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
}

best_model = None
best_score = -1
best_name = ""

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")

    labels = model.fit_predict(X_train)

    # Skip if only one cluster formed
    if len(set(labels)) < 2:
        print(f"{name} skipped (only one cluster formed)")
        continue

    sil_score = silhouette_score(X_train, labels)
    db_score = davies_bouldin_score(X_train, labels)

    print(f"{name} Silhouette Score: {sil_score}")
    print(f"{name} Davies-Bouldin Score: {db_score}")

    joblib.dump(model, f"models/{name}_clustering.pkl")

    if sil_score > best_score:
        best_score = sil_score
        best_model = model
        best_name = name

# Save best model
if best_model is not None:
    joblib.dump(best_model, "models/best_clustering_model.pkl")
    print(f"\nBest Clustering Model: {best_name}")
    print("Best Model Saved Successfully!")
else:
    print("No valid clustering model found.")