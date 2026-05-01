from sklearn.cluster import KMeans
import numpy as np           

def run_kmeans(X, k):
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)
    return cluster_labels
