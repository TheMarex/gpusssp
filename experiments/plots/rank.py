import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_rank_boxplot(variant_dfs, title):
    df_plot = pd.concat([df.assign(variant=name) for name, df in variant_dfs.items()], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_plot, x='rank', y='time', hue='variant', ax=ax)
    ax.set_xlabel('Query Rank')
    ax.set_ylabel('Time (μs)')
    plt.yscale('log')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
