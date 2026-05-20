"""
Small heatmaps I can run for each biomarker individually.
--------------------------------------------------------------------------------
`src.utils.figures.general.plot_correlations`

"""
import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn           as sns

from typing      import Optional, List
from scipy.stats import pearsonr

import math

# ================================================================================
# Plot correlation heatmaps with significance stars
# ================================================================================
def plot_correlations_by_dataset(
    df        : pd.DataFrame,
    cols      : Optional[List[str]] = None,
    group_col :               str   = "srcID",
    groups    : Optional[List[str]] = None, #["Hand", "Azure", "Whisper", "WhisperTiny", ], # "Pitt"
    do_all    : bool                = True,
    title     :               str   = None,
    font_size :               int   = 12,
):
    if cols   is None: cols   = ["mmse", "cls", "reg", "age", "sex"]
    if groups is None: groups = list(df[group_col].dropna().unique())

    font = "Times New Roman"
    sns.reset_defaults()

    # --------------------------------------------------------------------------------
    # Correlation + p-values + label matrix helper function 
    # --------------------------------------------------------------------------------
    def corr_with_p(df_subset: pd.DataFrame):
        df_subset = df_subset[cols].dropna()
        corr  = df_subset.corr(method="pearson")
        pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if i <= j:
                    r, p = pearsonr(df_subset[a], df_subset[b])
                    pvals.loc[a, b] = p
                    pvals.loc[b, a] = p

        def star(p):
            if p < 0.001: return "***"
            if p < 0.010: return "**"
            if p < 0.050: return "*"
            return ""

        labels = corr.round(2).astype(str)
        for i in range(len(cols)):
            for j in range(len(cols)):
                labels.iloc[i, j] = labels.iloc[i, j] + star(pvals.iloc[i, j])

        return corr, labels

    # --------------------------------------------------------------------------------
    # Figure layout (overall + one per group)
    # --------------------------------------------------------------------------------
    n_panels = (1 if do_all else 0) + len(groups)
    n_cols   = 2
    n_rows   = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # Pad an extra empty graph
    if n_panels % 2 == 1: groups += ["empty"]
    
    # --------------------------------------------------------------------------------
    # First panel includes all data concatenated
    # --------------------------------------------------------------------------------
    corr_all, labels_all = corr_with_p(df)

    if do_all:
        mask = np.triu(np.ones_like(corr_all, dtype=bool))
        sns.heatmap(
            corr_all, mask=mask, annot=labels_all, fmt="", cmap="vlag",
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
            annot_kws={"fontsize": font_size, "fontfamily": font},
            cbar_kws={"shrink": .8, "label": "Pearson r"},
            ax=axes[0],
        )
        axes[0].set_title("All datasets")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].tick_params(axis="y", rotation=0)

    # --------------------------------------------------------------------------------
    # Remaining panels: each dataset
    # --------------------------------------------------------------------------------
    start_idx = 1 if do_all else 0
    for ax, ds in zip(axes[start_idx:], groups):
        df_ds = df[df[group_col] == ds].copy()
        if df_ds[cols].dropna().empty:
            ax.axis("off")
            ax.set_title(f"{ds} (no data)")
            continue

        corr_ds, labels_ds = corr_with_p(df_ds)
        mask_ds = np.triu(np.ones_like(corr_ds, dtype=bool))

        sns.heatmap(
            corr_ds, mask=mask_ds, annot=labels_ds, fmt="", cmap="vlag",
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
            annot_kws={"fontsize": font_size, "fontfamily": font},
            cbar_kws={"shrink": .8, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(ds)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    # --------------------------------------------------------------------------------
    # Title and display
    # --------------------------------------------------------------------------------
    if title == None: title = "Correlation Matrices by Dataset"
    title += "\n(stars = p < .05 / .01 / .001)"
    
    plt.suptitle(title, y=1.02, fontsize=font_size, fontfamily=font)
    plt.tight_layout()
    plt.show()





# ================================================================================
# Plot correlation heatmaps with significance stars
# ================================================================================
def plot_correlations(df, cols=None, x_rot=30):
    df = df.copy()
    
    # --------------------------------------------------------------------------------
    # Get correlations & significance
    # --------------------------------------------------------------------------------
    if not cols: cols = ["mmse", "cls", "reg", "age", "sex"]
    corr = df[cols].corr(method="pearson")

    # Get p-values for significance
    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i <= j:
                r, p = pearsonr(df[a], df[b])
                pvals.loc[a, b] = p
                pvals.loc[b, a] = p

    def star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # --------------------------------------------------------------------------------
    # Format labels using correlations + stars
    # --------------------------------------------------------------------------------
    labels = corr.round(2).astype(str)
    for i in range(len(cols)):
        for j in range(len(cols)):
            labels.iloc[i, j] = labels.iloc[i, j] + star(pvals.iloc[i, j])

    # --------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------
    font = "Times New Roman"
    #sns.set_theme(style="white", context="talk")
    sns.reset_defaults()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    #fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr, mask=mask, annot=labels, fmt="", cmap="vlag",
        vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
        annot_kws={"fontsize": 12, "fontfamily": font},
        cbar_kws={"shrink": .8, "label": "Pearson r"}, ax=ax
    )

    ax.set_title("Correlation Matrix (Pearson r, stars = p < .05 / .01 / .001)", pad=12)
    plt.xticks(rotation=x_rot); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

