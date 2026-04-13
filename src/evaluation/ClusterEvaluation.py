from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_clusters(X, cluster_labels, true_labels, metric='euclidean'):
    X_input = X.toarray() if hasattr(X, 'toarray') else X
    sil = silhouette_score(X_input, cluster_labels,
                           sample_size=2000, metric=metric)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return sil, ari, nmi