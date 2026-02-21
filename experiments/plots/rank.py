from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_rank_boxplot(
    variant_dfs,
    title,
    variant_filter=None,
    output_path=None,
    show=True,
):
    filtered_items = [
        df.assign(variant=name)
        for name, df in variant_dfs.items()
        if variant_filter is None or variant_filter in name
    ]
    if not filtered_items:
        raise ValueError("No variants available for rank boxplot")
    df_plot = pd.concat(filtered_items, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_plot, x="rank", y="time", hue="variant", ax=ax)
    ax.set_xlabel("Query Rank")
    ax.set_ylabel("Time (μs)")
    plt.yscale("log")
    ax.set_title(title)
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
