# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Dimensionality reduction with pca
def reduce_dimensions_withPCA(X, n_components=2, random_state=42):
    X_dense = X.toarray() if hasattr(X, 'toarray') else np.array(X)
    extra = {}

    reducer = PCA(n_components=n_components, random_state=random_state)
    X_2d = reducer.fit_transform(X_dense)
    extra['explained_variance'] = reducer.explained_variance_ratio_
    extra['total_variance'] = sum(reducer.explained_variance_ratio_)
    print(f"PCA — Explained variance: PC1={extra['explained_variance'][0]:.4f}, "
              f"PC2={extra['explained_variance'][1]:.4f}, "
              f"Total={extra['total_variance']:.4f}")

    return X_2d, extra

# Visualize the embeddings and vectors
def visualize_embeddings(X, df, label_col='Ticket Type',
                         title=None, ax=None, figsize=(10, 7), alpha=0.4,
                         point_size=8):
    X_2d, extra = reduce_dimensions_withPCA(X)
    labels = df[label_col].values

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    palette = sns.color_palette('tab10', n_colors=len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[palette[i]], label=label,
            alpha=alpha, s=point_size, edgecolors='none'
        )

    if title is None:
        title = f"{label_col}"

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Component 1')
    ax.set_ylabel(f'Component 2')
    ax.legend(fontsize=9, markerscale=3, loc='best')
    ax.grid(True, alpha=0.2)

    return X_2d, extra

# Compare the visualization between the vectors and embeddings
def compare_representations(X_bow, X_tfidf, X_sbert, X_skipgram,df,
                            label_col, method="pca"):

    fig, axes = plt.subplots(1, 4, figsize=(32, 7))

    representations = [
        (X_bow, 'BoW'),
        (X_tfidf, 'TF-IDF'),
        (X_sbert, 'SBERT'),
        (X_skipgram, "Skipgram")
    ]

    for ax, (X, name) in zip(axes, representations):
        visualize_embeddings(
            X, df,
            label_col=label_col,
            title=f'{name} — {method.upper()}',
            ax=ax
        )

    fig.suptitle(
        f'Representation Comparison — {method.upper()} coloured by {label_col}',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()