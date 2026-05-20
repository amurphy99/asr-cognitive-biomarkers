"""
Generate correletion tables for each biomarker vs. MMSE across ASR sources.
--------------------------------------------------------------------------------
`src.utils.analysis.meta_correlation_table`

"""
import numpy  as np
import pandas as pd

from IPython.display import display

# Project code
from ..logging.logging import RESET, BOLD, H_LINE_1A, H_LINE_1B, H_LINE_2A, H_LINE_2B
from ..config          import BIOMARKER_COLS, RENAME_MAP, BIOMARKER_NAMES
from  .meta_analysis   import get_mmse_corr_by_src, corr_with_wer


# ================================================================================
# Build and display the full biomarker vs. MMSE correlation table set
# ================================================================================
def build_meta_correlation_table(
    res_base        : pd.DataFrame,             # Combined results DataFrame (pre-rename)
    wer_df          : pd.DataFrame,             # WER DataFrame with 'srcID', 'pID', 'WER' columns
    biomarker_cols  : tuple = BIOMARKER_COLS,   # Biomarker column names to include
    rename_map      : dict  = RENAME_MAP,       # Column rename map applied to res_base before computing
    biomarker_names : dict  = BIOMARKER_NAMES,  # Display name mapping for the meta-correlation table index
) -> tuple:
    """
    Full processing for the final biomarker vs. MMSE correlations & WER 
    meta-correlations.
    """
    # --------------------------------------------------------------------------------
    # Rename biomarker columns
    # --------------------------------------------------------------------------------
    # Make a working copy of the combined results and rename some of the columns
    res_renamed = res_base.copy()
    res_renamed = res_renamed.rename(columns=rename_map)

    # --------------------------------------------------------------------------------
    # MMSE correlation per source
    # --------------------------------------------------------------------------------
    corr_table1 = get_mmse_corr_by_src(
        res_renamed,
        biomarker_cols = biomarker_cols,
        group_col      = "srcID",
        include_all    = False,
    )

    # --------------------------------------------------------------------------------
    # Add average WER per transcript source to the correlation table
    # --------------------------------------------------------------------------------
    wer_col = []
    for key in corr_table1.index:
        subset_w: pd.DataFrame = wer_df[wer_df["srcID"] == key]
        wer_col.append(subset_w["WER"].mean() if not subset_w.empty else None)

    # Add the column
    corr_table = corr_table1.copy()
    corr_table.insert(0, "WER", wer_col)
    corr_table = corr_table.sort_values(by="WER", ascending=True)

    # --------------------------------------------------------------------------------
    # Display correlation table (not the meta-correlations yet)
    # --------------------------------------------------------------------------------
    # Make a display copy
    display_corr = corr_table.copy()

    # Insert vertical separators
    display_corr.insert(7, "|",    "|")
    display_corr.insert(5, "| ",   "|")
    display_corr.insert(3, "|  ",  "|")
    display_corr.insert(1, "|   ", "|")

    # Display the new table with p-values
    print(f"{H_LINE_2A}{BOLD} Pearson Correlation Table{RESET} (biomarkers vs. MMSE){H_LINE_2B}")
    display(display_corr.round(4))

    # ================================================================================
    # Create the **meta-correlation** table (Pearson r & Fisher-z)
    # ================================================================================
    # Fisher-Z transformation
    fisher_corr_table = corr_table.copy()
    for col in list(biomarker_cols):
        fisher_corr_table[col] = np.arctanh(fisher_corr_table[col].values)

    # --------------------------------------------------------------------------------
    # Meta-correlation: biomarker r vs. WER
    # --------------------------------------------------------------------------------
    # Exclude the "Hand Annotated" transcripts where WER == 0
    base_rm_hand =        corr_table[       corr_table["WER"] > 0].copy()
    fish_rm_hand = fisher_corr_table[fisher_corr_table["WER"] > 0].copy()

    # Biomarker meta-correlation
    base_wer_corrs = corr_with_wer(base_rm_hand, cols=biomarker_cols)
    fish_wer_corrs = corr_with_wer(fish_rm_hand, cols=biomarker_cols)

    # --------------------------------------------------------------------------------
    # Display meta-correlation tables | NOTE: Now display a merged table instead
    # --------------------------------------------------------------------------------
    #print(f"{BOLD}Pearson Meta-Correlation Table{RESET}")
    #display(base_wer_corrs)

    #print(f"{BOLD}Fisher-z Meta-Correlation Table{RESET}")
    #display(fish_wer_corrs)

    # --------------------------------------------------------------------------------
    # Merge and display meta-correlation tables side by side
    # --------------------------------------------------------------------------------
    # Merge the two tables for display purposes
    combined_wer_corrs = pd.concat({"Pearson r": base_wer_corrs, "Fisher-z": fish_wer_corrs}, axis=1)

    # Insert vertical separators
    combined_wer_corrs.insert(len(   base_wer_corrs.columns),  ("", "|"), "|"                       )  # Middle
    combined_wer_corrs.insert(0,                               ("", "|"), "|", allow_duplicates=True)  # Left
    combined_wer_corrs.insert(len(combined_wer_corrs.columns), ("", "|"), "|", allow_duplicates=True)  # Right

    # Use the full biomarker names for display
    combined_wer_corrs = combined_wer_corrs.rename(index=biomarker_names)

    # Display with p-values
    print(f"{H_LINE_1A}{BOLD} Meta-Correlation Table{RESET} (Pearson r & Fisher-z){H_LINE_1B}")
    display(combined_wer_corrs.round(4))

    # Return the tables for use elsewhere
    return corr_table, fisher_corr_table, base_wer_corrs, fish_wer_corrs

