import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_histogram(variant_dfs, title, filter=None):
    if filter is not None:
        variant_dfs = {name: df for name, df in variant_dfs.items() if filter in name}
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_times = np.concatenate([df['time'].values for df in variant_dfs.values()])
    p99 = np.percentile(all_times, 99)
    all_times = all_times[all_times <= p99]
    bins = np.histogram_bin_edges(all_times, bins=50)
    
    for name, df in variant_dfs.items():
        sns.histplot(df['time'], label=name, alpha=0.5, ax=ax, bins=bins)
    
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
