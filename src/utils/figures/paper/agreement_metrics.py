"""
Calculate & plot 3 agreement metrics between manual and ASR-derived biomarkers.
--------------------------------------------------------------------------------
`src.utils.figures.paper.agreement_metrics`

"""
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns

from matplotlib.axes import Axes
from scipy.stats     import pearsonr, spearmanr, zscore

# From this project
from ...config import FONT_FAMILY, SRC_MAP_SHORT, BIOMARKER_MAP_SHORT


# ================================================================================
# Calculate three agreement metrics between manual and ASR-derived biomarkers
# ================================================================================
def calculate_agreement_metrics(
    df         : pd.DataFrame,   # 'res_renamed'
    biomarkers : list,           # Biomarkers to do this for
    manual_id  : str = "Hand",   # Reference transcript source
    id_col     : str = "pID",    # Column denoting the participant ID
    src_col    : str = "srcID",  # Column denoting the transcript source (ASR method)
    verbose    : int = 1,
) -> pd.DataFrame:
    """
    Calculates three agreement metrics between manual and ASR-derived biomarkers:
      1. Pearson r       (linear correlation)
      2. Spearman rho    (rank correlation; robust to outliers)
      3. MAD Z-score     (mean absolute difference of z-normalized values)
 
    Returns a long-form DataFrame with one row per (Biomarker, ASR_System) pair.
    """
    # Get list of ASR systems (excluding manual)
    asr_systems  = [x for x in df[src_col].unique() if x != manual_id]
 
    # Print which is being used as reference
    if verbose >= 1: print(f"Comparing '{manual_id}' against {len(asr_systems)} ASR systems:\n  {asr_systems}")
 
    # --------------------------------------------------------------------------------
    # For each biomarker type...
    # --------------------------------------------------------------------------------
    results_list = []
    for bm in biomarkers:
        # Pivot data to align Manual and ASR rows by pID
        pivot = df.pivot_table(index=id_col, columns=src_col, values=bm)
 
        # We need the "Manual" (reference) column to exist
        if manual_id not in pivot.columns:
            print(f"Skipping {bm}: '{manual_id}' not found.")
            continue
 
        # --------------------------------------------------------------------------------
        # For each ASR system
        # --------------------------------------------------------------------------------
        for system in asr_systems:
            # Make sure there are enough samples (there always should be anyways)
            if system not in pivot.columns: continue
                
            # Get paired data (dropping missing values)
            pair = pivot[[manual_id, system]].dropna()
            if len(pair) < 3: continue  # need at least 3 points for correlation
 
            # Get the manual and ASR system values
            x_man = pair[manual_id]
            y_asr = pair[system]

            # --------------------------------------------------------------------------------
            # Calculate the three metrics
            # --------------------------------------------------------------------------------
            # 1) Pearson correlation
            r,   p_r   = pearsonr (x_man, y_asr)
            
            # 2) Spearman correlation
            rho, p_rho = spearmanr(x_man, y_asr)
 
            # 3) MAD Z-score
            z_man = zscore(x_man)
            z_asr = zscore(y_asr)
            mad_z = np.mean(np.abs(z_man - z_asr))
 
            results_list.append({
                "Biomarker"    : bm,
                "ASR_System"   : system,
                "Pearson_r"    : r,
                "Pearson_p"    : p_r,
                "Spearman_rho" : rho,
                "Spearman_p"   : p_rho,
                "MAD_Z"        : mad_z,
            })
 
    # Return results DataFrame
    return pd.DataFrame(results_list)


# ================================================================================
# Agreement Metric Heatmaps
# ================================================================================
def plot_agreement_heatmaps(
    agreement_df  : pd.DataFrame,       # Output of calculate_agreement_metrics
    wer_df        : pd.DataFrame,       # WER DataFrame with 'srcID' and 'WER' columns
    src_map       : dict = None,        # Short display name map for ASR systems
    biomarker_map : dict = None,        # Short display name map for biomarkers
    font          : str  = FONT_FAMILY, # Font family for all text
) -> None:
    """
    Three side-by-side heatmaps showing agreement between manual and ASR-derived
    biomarkers: Pearson r, Spearman rho, and MAD Z-score. ASR systems are sorted
    by ascending mean WER.
    """
    if src_map       is None:       src_map =       SRC_MAP_SHORT
    if biomarker_map is None: biomarker_map = BIOMARKER_MAP_SHORT
 
    # --------------------------------------------------------------------------------
    # 1) Prepare display data
    # --------------------------------------------------------------------------------
    df_display = agreement_df.copy()
    df_display = df_display[df_display["Biomarker"] != "Gram_r"]
    
    # Rename biomarkers & ASR systems
    df_display["Biomarker"]  = df_display["Biomarker" ].replace(biomarker_map)
    df_display["ASR_System"] = df_display["ASR_System"].replace(      src_map)
 
    # --------------------------------------------------------------------------------
    # 2) Sort transcript sources by WER (lowest-to-highest)
    # --------------------------------------------------------------------------------
    # Calculate mean WER per system
    wer_sorted          = wer_df.copy()
    wer_sorted["srcID"] = wer_sorted["srcID"].replace(src_map)
    system_wer          = wer_sorted.groupby("srcID")["WER"].mean().sort_values(ascending=True)
 
    # Filter to only include systems present in our agreement data
    valid_systems  = set(df_display["ASR_System"].unique())
    sorted_systems = [s for s in system_wer.index if s in valid_systems]
    
    # Append any systems missing from WER data
    remaining      = [s for s in valid_systems if s not in sorted_systems]
    sorted_systems.extend(remaining)
    if remaining: print(f"WARNING: Systems missing WER data appended to end: {remaining}")
 
    # --------------------------------------------------------------------------------
    # 3) Pivot data
    # --------------------------------------------------------------------------------
    # Create pivots and enforce the sorted order
    common     = dict(index="ASR_System", columns="Biomarker")
    pivot_r    = df_display.pivot(values="Pearson_r",    **common).reindex(sorted_systems)
    pivot_rho  = df_display.pivot(values="Spearman_rho", **common).reindex(sorted_systems)
    pivot_mad  = df_display.pivot(values="MAD_Z",        **common).reindex(sorted_systems)
 
    # --------------------------------------------------------------------------------
    # 4) Styling
    # --------------------------------------------------------------------------------
    TITLE_SIZE = 28
    LABEL_SIZE = 24
    ANNOT_SIZE = 32
 
    # Shared heatmap settings
    heatmap_kws = {
        "annot"    : True,
        "fmt"      : ".2f",
        "cbar"     : False,
        "annot_kws": {"size": ANNOT_SIZE, "fontfamily": font},
    }
 
    # ================================================================================
    # 5) Plot
    # ================================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 9), sharey=True)
 
    # Plot A: Pearson r (Linear)
    sns.heatmap(pivot_r,   cmap="Blues",  ax=axes[0], **heatmap_kws)
    axes[0].set_title("Pearson Correlation ($r$)\nLinear Relationship (higher is better)",
                      fontsize=TITLE_SIZE, fontfamily=font, fontweight="bold", pad=20)
 
    # Plot B: Spearman rho (Rank)
    sns.heatmap(pivot_rho, cmap="Greens", ax=axes[1], **heatmap_kws)
    axes[1].set_title("Spearman Correlation ($\\rho$)\nRank Ordering (higher is better)",
                      fontsize=TITLE_SIZE, fontfamily=font, fontweight="bold", pad=20)
 
    # Plot C: MAD Z-Score (Error)
    sns.heatmap(pivot_mad, cmap="Reds",   ax=axes[2], **heatmap_kws)
    axes[2].set_title("Mean Absolute Diff (Z-Score)\nDistribution Error (lower is better)",
                      fontsize=TITLE_SIZE, fontfamily=font, fontweight="bold", pad=20)
 
    # --------------------------------------------------------------------------------
    # Styling & Labels
    # --------------------------------------------------------------------------------
    for ax in axes:
        ax: Axes = ax  # (just for the syntax highlighting)

        # No axis labels
        ax.set_xlabel(""); ax.set_ylabel("")
        
        # Configure tick parameters
        ax.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
        
        # Rotate x-axis tick labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha="center", fontsize=LABEL_SIZE,   fontfamily=font)
        
        # Rotate y-axis labels (applied to all, visible on left-most)
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right",  fontsize=LABEL_SIZE,   fontfamily=font)
 
    # Overall title
    plt.suptitle("Agreement Between Reference-derived & ASR-derived Biomarkers", y=1.00, fontsize=32, fontfamily=font, fontweight="bold")
 
    # Display
    plt.tight_layout()
    plt.show()

