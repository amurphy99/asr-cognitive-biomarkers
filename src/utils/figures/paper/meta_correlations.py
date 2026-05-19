"""
Plot the meta-correlations between biomarkers, MMSE scores, and WER.
--------------------------------------------------------------------------------
`src.utils.figures.paper.meta_correlations`

First, the Pearson correlation between biomarker scores and MMSE is taken for 
various transcripts (manually annotated, ASR methods). The resulting correlation
coefficients are then Fisher-Z transformed before we take the Pearson 
correlation of those coefficients with their corresponding transcript sources
average word error rate (WER).

TODO: Add a line to save the figure with a high DPI... (probably do the same
      for all of these figures...)
"""
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns

from matplotlib.axes import Axes

# From this project
from ...config import FONT_FAMILY, BIOMARKER_NAMES


# ================================================================================
# Meta-correlation Plots
# ================================================================================
def plot_meta_correlations(
    corr_table      : pd.DataFrame, # Pearson correlation table for biomarkers vs. MMSE across sources
    biomarkers      : list = None,  # List of biomarker columns to include in the plot
    biomarker_names : dict = None,  # List of full biomarker names for plotting
    font_family     : str  = FONT_FAMILY, # Font to use across the figure
) -> None:
    """
    Plots scatter + regression for correlation-vs-WER (left axis, blue) and
    p-value-vs-WER (right axis, red) for each biomarker. Rows with WER == 0
    (i.e. Hand/manual transcripts w WER==0) are excluded.
 
    Parameters
    ----------
    corr_table      : output of compute_meta_correlation_table (raw or Fisher-z)
    biomarkers      : list of column names to plot (default: AltGram, Perplex, PragImp)
    biomarker_names : display name mapping
    font_family     : 
    """
    if biomarkers      is None: biomarkers = ["AltGram", "Perplex", "PragImp"]
    if biomarker_names is None: biomarker_names = BIOMARKER_NAMES
 
    # Exclude Hand-annotated transcriptions (0 WER)
    rm_hand = corr_table[corr_table["WER"] > 0].copy()
 
    # --------------------------------------------------------------------------------
    # Styling
    # --------------------------------------------------------------------------------
    TITLE_STYLE  = {"fontsize": 36, "fontfamily": font_family, "fontweight": "normal"}
    Y_AXIS_LABEL = {"fontsize": 32, "fontfamily": font_family, "fontweight": "normal"}
    X_AXIS_LABEL = {"fontsize": 28, "fontfamily": font_family, "fontweight": "normal"}
    TICK_PARAMS  = {"labelsize": 20, "length": 6, "width": 2}
    LEGEND_STYLE = {"size": 28, "family": font_family}
    KWS_SCATTER  = {"s": 60, "marker": "o"}
 
    # --------------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(biomarkers), figsize=(22, 6), sharey=True)
    axes: np.ndarray[Axes] = axes.flatten()
 
    for i, bm in enumerate(biomarkers):
        ax1: Axes = axes[i]
        ax2: Axes = ax1.twinx()
        p_col     = f"{bm}_p"
 
        # --------------------------------------------------------------------------------
        # Left Axis: Correlation (Blue)
        # --------------------------------------------------------------------------------
        color = "tab:blue"
        sns.regplot(data=rm_hand, x="WER", y=bm, ax=ax1, color=color,
            label       = "Correlation ($r$)",
            scatter_kws = KWS_SCATTER,
            line_kws    = {"linewidth": 3},
        )
        
        # Labels & Tick Settings (some stuff done always, some only on the edge plots)
        if i == 0: ax1.set_ylabel("Correlation ($r$)", color=color, **Y_AXIS_LABEL)
        else:      ax1.set_ylabel(""); ax1.tick_params(axis="y", left=False)
        ax1.tick_params(axis="y", color=color, labelcolor=color, **TICK_PARAMS)
        
        # X-Axis
        ax1.set_xlabel("Word Error Rate (WER)", **X_AXIS_LABEL)
        ax1.tick_params(axis="x", **TICK_PARAMS)
 
        # --------------------------------------------------------------------------------
        # Right Axis: P-value (Red)
        # --------------------------------------------------------------------------------
        color = "tab:red"
        sns.regplot(data=rm_hand, x="WER", y=p_col, ax=ax2, color=color,
            label       = "p-value",
            scatter_kws = KWS_SCATTER,
            line_kws    = {"linestyle": "--", "linewidth": 3},
        )
        
        # Labels & Tick Settings (some stuff done always, some only on the edge plots)
        if i == len(biomarkers)-1: ax2.set_ylabel("$p$-value", color=color, **Y_AXIS_LABEL)
        else:                      ax2.set_ylabel(""); ax2.tick_params(axis="y", labelright=False, right=False)
        ax2.tick_params(axis="y", color=color, labelcolor=color, **TICK_PARAMS)
        
        # P-value threshold (dotted line at p == 0.05)
        ax2.axhline(0.05, color="black", linestyle=":", linewidth=2, alpha=0.8, label="p=0.05")
        
        # Shared grid and title
        ax2.grid(True, linestyle=":", alpha=0.99)
        ax1.set_title(biomarker_names.get(bm, bm), **TITLE_STYLE, pad=15)
 
    # --------------------------------------------------------------------------------
    # Combined Legend
    # --------------------------------------------------------------------------------
    # Take labels from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels() # Blue line
    lines_2, labels_2 = ax2.get_legend_handles_labels() # Red line + Black threshold

    # Combine them on one legend
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
        loc            = "lower center",
        bbox_to_anchor = (0.5, -0.125),  # Pushes legend below the X-axis labels
        ncol           = 3,              # Spread items horizontally
        frameon        = True,
        fancybox       = True,
        shadow         = False,
        markerscale    = 2.0,
        prop           = LEGEND_STYLE,
    )
 
    # Display
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Tight layout to leave room at the bottom for the legend
    plt.subplots_adjust(wspace=0.05)       # Squeeze the plots together horizontally
    plt.show()

