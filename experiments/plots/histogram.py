from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_histogram(
    variant_dfs, title, variant_filter=None, output_path=None, show=True
):
    if variant_filter is not None:
        variant_dfs = {
            name: df for name, df in variant_dfs.items() if variant_filter in name
        }
    if not variant_dfs:
        raise ValueError("No variants available for histogram plot")
    fig, ax = plt.subplots(figsize=(10, 6))

    all_times = np.concatenate([df["time"].values for df in variant_dfs.values()])
    p99 = np.percentile(all_times, 99)
    all_times = all_times[all_times <= p99]
    bins = np.histogram_bin_edges(all_times, bins=50)

    for name, df in variant_dfs.items():
        sns.histplot(df["time"], label=name, alpha=0.5, ax=ax, bins=bins)

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        saved_path = output_path

    if show:
        plt.show()
    plt.close(fig)

    return saved_path
