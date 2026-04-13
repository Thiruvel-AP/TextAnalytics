import matplotlib.pyplot as plt
import pandas as pd

def visualize_cluster_stability(df, algorithm='KMeans', 
                                   metrics=['Silhouette', 'ARI (Type)', 'NMI (Type)']):
    # Filter for the specific algorithm
    plot_df = df[df['Algorithm'] == algorithm].copy()
    
    # Identify unique representations
    representations = plot_df['Representation'].unique()
    
    num_metrics = len(metrics)
    num_reps = len(representations)
    
    # Create the grid: Rows = Metrics, Columns = Representations
    fig, axes = plt.subplots(num_metrics, num_reps, 
                             figsize=(5 * num_reps, 4 * num_metrics), 
                             squeeze=False)

    # Define colors for the metrics to make them distinct
    metric_colors = {
        'Silhouette': 'royalblue',
        'ARI (Type)': 'forestgreen',
        'NMI (Type)': 'darkorange',
        'ARI (Subject)': 'crimson',
        'NMI (Subject)': 'purple'
    }

    for row_idx, metric in enumerate(metrics):
        for col_idx, rep_name in enumerate(representations):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this specific combination
            data = plot_df[plot_df['Representation'] == rep_name].sort_values('k')
            
            k_values = data['k'].values
            scores = data[metric].values
            color = metric_colors.get(metric, 'blue')
            
            # Plotting the line and points
            ax.plot(k_values, scores, marker='o', linewidth=2, markersize=7, color=color)
            ax.fill_between(k_values, scores, alpha=0.1, color=color)
            
            # Annotate each point with its value
            for k, score in zip(k_values, scores):
                ax.annotate(f'{score:.3f}', (k, score), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
            
            # Formatting the subplot
            if row_idx == 0:
                ax.set_title(f'Representation: {rep_name}', fontsize=14, pad=20, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Number of clusters (k)')
            ax.set_xticks(k_values)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set consistent Y-axis limits for normalized scores
            ax.set_ylim(min(0, scores.min() - 0.05), max(1, scores.max() + 0.1))

    plt.suptitle(f'Comprehensive Clustering Analysis ({algorithm})', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()