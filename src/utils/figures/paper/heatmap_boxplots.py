"""
Summarize biomarker performance across transcript conditions.
--------------------------------------------------------------------------------
`src.utils.figures.paper.heatmap_boxplots`

NOTE: Figure 1: Biomarker Performance Across Transcript Conditions

"Biomarker performance across transcript conditions. Left: Pearson correlation 
between each biomarker and MMSE for Manual and ecological ASR transcript sources
(systems ordered by increasing WER). Right: biomarker score distributions for
Control vs. ProbableAD participants across transcript sources. Biomarker scores
are such that higher values indicate greater impairment, and associations with
MMSE are negative. All reported correlations are statistically significant 
(p < 0.001)."

TODO: Maybe this should also take the corr_table as input...
"""
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns

from matplotlib.lines import Line2D
from scipy.stats      import pearsonr

# From this project
from ...config import FONT_FAMILY, BIOMARKER_NAMES, SRC_MAP_SHORT, RENAME_MAP, BIOMARKER_MAP_SHORT


# ================================================================================
# Biomarker Performance Across Transcript Conditions
# ================================================================================
def plot_heatmap_and_boxplots(
    res_base        : pd.DataFrame,        # Combined results DataFrame (pre-rename)
    wer_df          : pd.DataFrame,        # WER DataFrame with 'srcID', 'pID', 'WER' columns
    src_map_short   : dict = None,         # Short display names for ASR systems
    biomarker_names : dict = None,         # Long display names for biomarkers (used in boxplot titles)
    rename_map      : dict = None,         # Column rename map applied to res_base before plotting
    font            : str  = FONT_FAMILY,  # Font family for all text
) -> None:
    """
    Two-panel figure (for paper):
      Left  — Heatmap of Pearson r between each biomarker and MMSE, per ASR system.
      Right — Boxplots of each biomarker split by diagnosis (Control vs ProbableAD).
 
    Both panels share the same WER-sorted y-axis order.
    """
    # Defaults
    if src_map_short   is None: src_map_short   = SRC_MAP_SHORT
    if biomarker_names is None: biomarker_names = BIOMARKER_NAMES
    if rename_map      is None: rename_map      = RENAME_MAP
    
    # Sizing constants
    MAIN_TITLE_SIZE = 32
    SUB_TITLE_SIZE  = 30
    LABEL_SIZE      = 28
    TICK_SIZE       = 24
    ANNOT_SIZE      = 32
 
    # --------------------------------------------------------------------------------
    # 1) Prepare a working copy of the data
    # --------------------------------------------------------------------------------
    res_2 = res_base.copy()
    res_2["srcID"] = res_2["srcID"].replace({"Hand": "Manually Annotated", "WhisperTiny": "Whisper-Tiny"})
    res_2 = res_2.rename(columns={**rename_map, "mmse": "MMSE"})
 
    biomarkers_right = ["AltGram", "Perplex", "PragImp"]
 
    # --------------------------------------------------------------------------------
    # 2) Sort transcript sources by WER (lowest-to-highest)
    # --------------------------------------------------------------------------------
    wer_temp = wer_df.copy()
    wer_temp["srcID"] = wer_temp["srcID"].replace(src_map_short)
    system_wer_order = (
        wer_temp.groupby("srcID")["WER"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
 
    # --------------------------------------------------------------------------------
    # 3) Left panel data prep (MMSE correlation heatmap)
    # --------------------------------------------------------------------------------
    # Rename columns and reorder sources
    df_calc = res_2.copy()
    df_calc["srcID"] = df_calc["srcID"].replace(src_map_short)
    valid_order_heatmap = [s for s in system_wer_order if s in df_calc["srcID"].unique()]
 
    # Get biomarker correlations with MMSE for each transcript source
    corr_data = []
    for src in valid_order_heatmap:
        subset = df_calc[df_calc["srcID"] == src]
        for bm in ["PragImp", "AltGram", "Perplex"]:
            valid = subset[[bm, "MMSE"]].dropna()
            if len(valid) > 2:
                r, _ = pearsonr(valid[bm], valid["MMSE"])
                corr_data.append({"ASR_System": src, "Biomarker": BIOMARKER_MAP_SHORT.get(bm, bm), "r": r})
 
    df_corr    = pd.DataFrame(corr_data)
    pivot_mmse = (
        df_corr.pivot(index="ASR_System", columns="Biomarker", values="r")
        .reindex(valid_order_heatmap)
    )
 
    # --------------------------------------------------------------------------------
    # 4) Right panel data prep (boxplots per biomarker)
    # --------------------------------------------------------------------------------
    CUSTOM_PALETTE = {"Control": "tab:blue", "ProbableAD": "tab:red"}
 
    df_plot = res_2.copy()
    df_plot["srcID"] = df_plot["srcID"].replace(src_map_short)
    valid_order_box = [s for s in system_wer_order if s in df_plot["srcID"].unique()]

 
    # ================================================================================
    # Build the final figure
    # ================================================================================
    fig = plt.figure(figsize=(22, 10))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1, 1.8], wspace=0.20, hspace=0.33)
 
    ax_left     = fig.add_subplot(gs[:, 0])
    axes_right  = [fig.add_subplot(gs[i, 1]) for i in range(3)]
 
    # --------------------------------------------------------------------------------
    # Left: Heatmap
    # --------------------------------------------------------------------------------
    sns.heatmap(
        pivot_mmse,
        annot     = True,
        fmt       = ".2f",
        cmap      = "RdBu_r",
        center    = 0,
        cbar      = False,
        ax        = ax_left,
        annot_kws = {"size": ANNOT_SIZE, "fontfamily": font},
    )
    # Remove axis labels & set tick size
    ax_left.set_xlabel(""); ax_left.set_ylabel("")
    ax_left.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    
    # Axis rotation
    plt.setp(ax_left.get_xticklabels(), rotation=30, ha="center", fontsize=LABEL_SIZE,   fontfamily=font)
    plt.setp(ax_left.get_yticklabels(), rotation= 0, ha="right",  fontsize=LABEL_SIZE-4, fontfamily=font)
 
    # --------------------------------------------------------------------------------
    # Right: Boxplots
    # --------------------------------------------------------------------------------
    # Outlier properties -> small, gray, semi-transparent
    flier_props = dict(marker="o", markerfacecolor="gray", markeredgecolor="none", alpha=0.5, markersize=4)
 
    # Boxplots for each biomarker type
    for i, bm in enumerate(biomarkers_right):
        ax      = axes_right[i]
        bm_data = df_plot[bm].dropna()
 
        # Create the boxplot
        sns.boxplot(data=df_plot, y="srcID", x=bm, hue="dx", order=valid_order_box, ax=ax,
            palette    = CUSTOM_PALETTE,
            linewidth  = 0.5,
            width      = 0.7,
            showfliers = True,
            flierprops = flier_props,  # apply visual noise reduction
        )
 
        # Trim x-axis margins
        low_lim  = np.percentile(bm_data,  1)
        high_lim = np.percentile(bm_data, 99)
        buffer   = (high_lim - low_lim) * 0.05
        ax.set_xlim(low_lim - buffer, high_lim + buffer)
        
        # Title
        ax.set_title(biomarker_names.get(bm, bm), 
                     fontsize=SUB_TITLE_SIZE, fontfamily=font, fontweight="normal", pad=10)
        
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="gray")
        ax.set_axisbelow(True)
 
        # Add a reference line at 0 for Perplexity
        if bm == "Perplex": ax.axvline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.8)
 
        # Tick labels
        ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE-8)
        ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE-1)
 
        if i == len(biomarkers_right) - 1: ax.set_xlabel("Biomarker Score", fontsize=LABEL_SIZE, fontfamily=font)
        else:                               ax.set_xlabel("")
 
        # Remove the default legend from all subplots (will add a global one manually)
        if ax.get_legend(): ax.get_legend().remove()
 
    # Tick font
    for ax in axes_right:
        plt.setp(ax.get_xticklabels(), fontfamily=font)
        plt.setp(ax.get_yticklabels(), fontfamily=font)
 
    # --------------------------------------------------------------------------------
    # Titles & Legend
    # --------------------------------------------------------------------------------
    # Left title
    fig.text(0.210, 0.935, "Clinical Validity\n(Correlation with MMSE)",
             ha="center", fontsize=MAIN_TITLE_SIZE, fontfamily=font, fontweight="bold")
 
    # Right title
    fig.text(0.620, 0.935, "Diagnostic Separation\n(Control vs. ProbableAD)",
             ha="center", fontsize=MAIN_TITLE_SIZE, fontfamily=font, fontweight="bold")
 
    # Legend to the right of the titles
    legend_elements = [
        Line2D([0], [0], color=CUSTOM_PALETTE["Control"   ], lw=4, label="Control"),
        Line2D([0], [0], color=CUSTOM_PALETTE["ProbableAD"], lw=4, label="ProbableAD"),
    ]
    fig.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.760, 1.035),
               frameon=True, prop={"family": font, "size": TICK_SIZE})
 
    # Display
    plt.subplots_adjust(top=0.88, left=0.08, right=0.92, bottom=0.08)
    plt.show()
 
