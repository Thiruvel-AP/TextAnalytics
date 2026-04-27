from sklearn.cluster import AgglomerativeClustering
import numpy as np           

def run_hac(X, k, linkage_method='ward'):
    X_input = X.toarray() if hasattr(X, 'toarray') else X
    hac = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    cluster_labels = hac.fit_predict(X_input)
    return cluster_labels