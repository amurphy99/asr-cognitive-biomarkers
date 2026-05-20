"""
Calculates feature/biomarker stability based on 2024 paper by Heitz et al.
--------------------------------------------------------------------------------
`src.utils.figures.paper.feature_stability`

NOTE: Heitz et al. 2024: https://aclanthology.org/2024.lrec-main.1386/ 


TODO: Mess around with the zero-crossing part of this, see if I can get
      something a bit better.
TODO: Change it to use the full biomarker names

"""
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn           as sns


# ================================================================================
# Calculates feature stability based on 2024 paper by Heitz et al.
# ================================================================================
def calculate_feature_stability(
    df         : pd.DataFrame,   # res_renamed
    biomarkers : list,           # Biomarkers to do this for
    manual_id  : str = "Hand",   # Reference transcript source
    id_col     : str = "pID",    # Column denoting the participant ID
    src_col    : str = "srcID",  # Column denoting the transcript source (ASR method)
    verbose    : int = 1,
) -> pd.DataFrame:
    """
    Calculates feature stability sj based on Heitz et al.
    sj = Mean across ASR systems of (Mean across samples of |Relative Difference|)
 
    NOTE: Biomarkers that cross zero (e.g. Perplex, Gram_r) can produce large
    stability scores because the denominator approaches zero near sign changes.
 
    Returns a DataFrame indexed by Biomarker, sorted ascending (lower = more stable).
    """
    # Create a dictionary of dataframes, one for each biomarker
    stability_scores = {}
    
    # Get list of all ASR systems (everything except manual)
    asr_systems = [x for x in df[src_col].unique() if x != manual_id]

    # Print which is being used as reference
    if verbose >= 1: print(f"Comparing '{manual_id}' against {len(asr_systems)} ASR systems:\n  {asr_systems}")
 
    for bm in biomarkers:
        # Create a subset for this biomarker with pID as index and srcID as columns
        pivot = df.pivot_table(index=id_col, columns=src_col, values=bm)
 
        # We need the "Manual" (reference) column to exist
        if manual_id not in pivot.columns:
            print(f"Skipping {bm}: '{manual_id}' column not found.")
            continue
        
        # Store the stability per ASR system (the inner mean in Eq 4)
        system_stabilities = []
        for system in asr_systems:
            if system not in pivot.columns: continue
     
            # Get Manual and ASR vectors (aligned by pID)
            pair_data = pivot[[manual_id, system]].dropna() # drop rows where either is NaN
            t_man     = pair_data[manual_id]
            t_asr     = pair_data[system]
 
            # Eq (3): Relative Difference d (modified => denominator uses |Man| + |ASR|)
            # d = (Man - ASR) / ((Man + ASR) / 2)
            # sMAPE-style modification leaves it the same for positive values
            # and helps keep results from zero-crossing values stay sensical.
            numerator   = t_man - t_asr
            denominator = (t_man.abs() + t_asr.abs()) / 2.0  # Original: (t_man + t_asr) / 2.0
            denominator = denominator.replace(0, np.nan)     # Only zero if both inputs are exactly 0
 
            d_values = numerator / denominator
    
            # Inner part of Eq (4): Mean of |d| for this system
            # (1/N) * Sum(|d_i|)
            s_k      = d_values.abs().mean()
            system_stabilities.append(s_k)
 
        # Outer part of Eq (4): Mean across ASR systems
        # (1/K) * Sum(s_k)
        if system_stabilities: stability_scores[bm] = np.mean(system_stabilities)
        else:                  stability_scores[bm] = np.nan
 
    # Convert results to a readable DataFrame and sort
    results = pd.DataFrame(list(stability_scores.items()), columns=["Biomarker", "Stability (sj)"])
    return results.set_index("Biomarker").sort_values("Stability (sj)")


# ================================================================================
# Bar chart of feature stability scores (lower = more robust to ASR errors)
# ================================================================================
def plot_feature_stability(
    stability_df    : pd.DataFrame,
    title_size      : int = 14,
    axis_size       : int = 12,
    tick_label_size : int = 12,
    bar_label_size  : int = 12,
) -> None:
    plt.figure(figsize=(8, 4))
    
    # Basic bar plot
    ax = sns.barplot(
        x       = stability_df.index, 
        y       = "Stability (sj)", 
        data    = stability_df, 
        palette = "viridis",
        hue     = stability_df.index,
    )
    
    # Set y-axis to log scale
    ax.set_yscale("log")
    
    # Add real numbers above each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=bar_label_size)
    
    # Plot title
    plt.title(
        ("Biomarker Stability -- Heitz et al. (2024)\n"
        "Lower = More Robust to ASR Errors"), 
        fontsize=title_size,
    )
    
    # Axis labels
    plt.ylabel("Stability Score ($s_j$) -- Log Scale", fontsize=axis_size)
    plt.xlabel("Biomarker",                            fontsize=axis_size)

    # Adjust the size of the individual tick labels on the x axis
    ax.tick_params(axis="x", labelsize=tick_label_size)
    
    # Expand y-limit slightly so the top label doesn't get cut off
    ax.set_ylim(top=ax.get_ylim()[1] * 2) 
    
    # Display
    plt.show()

