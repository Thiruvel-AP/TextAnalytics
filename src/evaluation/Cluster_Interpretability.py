import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

def plot_cluster_top_words(X, cluster_labels, vectorizer, n_top=10, title_prefix="TF-IDF"):
    feature_names = vectorizer.get_feature_names_out()
    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    n_clusters = len(set(cluster_labels))
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 4.5))
    if n_clusters == 1: axes = [axes]
    for c in sorted(set(cluster_labels)):
        mask = cluster_labels == c
        cluster_mean = X_dense[mask].mean(axis=0)
        if hasattr(cluster_mean, 'A1'): cluster_mean = cluster_mean.A1
        top_idx = cluster_mean.argsort()[-n_top:][::-1]
        ax = axes[c]
        ax.barh(range(n_top), [cluster_mean[i] for i in top_idx],
                color=sns.color_palette('tab10')[c])
        ax.set_yticks(range(n_top))
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'Cluster {c} (n={mask.sum()})', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Mean {title_prefix}')
    fig.suptitle(f'{title_prefix} — Top {n_top} Words per Cluster (k={n_clusters})',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cluster_heatmaps(cluster_labels, df, title_prefix="TF-IDF"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    ct = pd.crosstab(cluster_labels, df['Ticket Type'])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{title_prefix} Clusters vs Ticket Type (Counts)', fontweight='bold')
    axes[0].set_ylabel('Cluster')
    sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1])
    axes[1].set_title(f'{title_prefix} Clusters vs Ticket Type (Proportions)', fontweight='bold')
    axes[1].set_ylabel('Cluster')
    plt.tight_layout()
    plt.show()

def plot_cluster_size(kmeans_vectors):
    fig, axes = plt.subplots(1, len(kmeans_vectors), figsize=(20, 10))
    for ax, (labels, name) in zip(axes, kmeans_vectors):
        unique, counts = np.unique(labels, return_counts=True)
        colors = sns.color_palette('tab10', len(unique))
        bars = ax.bar(unique, counts, color=colors)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count}\n({count/len(labels)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(f'{name} — Cluster Sizes (k=5)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Tickets')
    plt.tight_layout()
    plt.show()
