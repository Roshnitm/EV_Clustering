import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import joblib
from sklearn.metrics import silhouette_score
from utils.feature_extractor import extract_features

X_test, _ = extract_features("Dataset/test/test")

model = joblib.load("models/best_clustering_model.pkl")

labels = model.fit_predict(X_test)

if len(set(labels)) > 1:
    print("\nSilhouette Score:")
    print(silhouette_score(X_test, labels))
else:
    print("Only one cluster formed. Cannot compute silhouette score.")
