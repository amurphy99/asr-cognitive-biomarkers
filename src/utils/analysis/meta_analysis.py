"""
Helper functions for doing the biomarker/MMSE/WER meta-analysis.
--------------------------------------------------------------------------------
`src.utils.analysis.meta_analysis`

"""
import pandas as pd

from scipy.stats import pearsonr

# From this project
from ..config import BIOMARKER_COLS


# ================================================================================
# Calculate Pearson correlations between biomarkers and MMSE scores
# ================================================================================
def get_mmse_corr_by_src(
    df             : pd.DataFrame,
    biomarker_cols : tuple = ("PragImp", "AltGram", "Perplex"),
    group_col      : str   = "srcID", # How to group rows (e.g., by data source: "Hand", "Azure", ...)
    groups         : list  = None,    # List of groups to include in the output 
    include_all    : bool  = False,   # Do one where we combine all results
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per srcID showing the absolute Pearson
    correlation between MMSE and each biomarker column.

    Columns will be: srcID, [Biomarker], [Biomarker]_p, ...
    """
    # If no groups is provided, do all of the different sources
    if groups is None: groups = list(df[group_col].dropna().unique())
    
    # Results storage
    rows = []
 
    # --------------------------------------------------------------------------------
    # Inner helper that gets the Pearson correlation for all biomarkers
    # --------------------------------------------------------------------------------
    def _row_for_subset(name: str, subset: pd.DataFrame):
        subset = subset[["mmse", *biomarker_cols]].dropna()
        if len(subset) < 2: return None
 
        row = {"srcID": name}
        for col in biomarker_cols:
            r, p = pearsonr(subset["mmse"], subset[col])
            row[col]          = abs(r)  # magnitude of r
            row[f"{col}_p"]   = p
        return row
 
    # Add an extra one for every source combined (not really used)
    if include_all:
        r_all = _row_for_subset("ALL", df)
        if r_all is not None: rows.append(r_all)
 
    # Add a correlation row for each transcript source
    for g in groups:
        r_g = _row_for_subset(g, df[df[group_col] == g])
        if r_g is not None:  rows.append(r_g)
 
    # Return a DataFrame of the results
    out = pd.DataFrame(rows)
    return out.set_index("srcID").sort_index()


# ================================================================================
# Get correlation of each biomarker column with WER
# ================================================================================
def corr_with_wer(
    df      : pd.DataFrame,
    wer_col : str       = "WER",
    cols    : list[str] = BIOMARKER_COLS,
) -> pd.DataFrame:
    """
    Correlates each biomarker column with WER and returns a sorted DataFrame
    of (metric, r_vs_WER, p_vs_WER).
    """
    rows = []
    for col in cols:
        # Filter out any missing data
        sub = df[[wer_col, col]].dropna()
        if len(sub) > 1:
            r, p = pearsonr(sub[wer_col], sub[col])
            rows.append({"metric": col, "r_vs_WER": r, "p_vs_WER": p})
 
    out = pd.DataFrame(rows).set_index("metric")
    return out.sort_values("r_vs_WER")

