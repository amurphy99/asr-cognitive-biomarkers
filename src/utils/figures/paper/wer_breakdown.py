"""
Break down the differences in WER between diagnostic populations.
--------------------------------------------------------------------------------
`src.utils.figures.paper.wer_breakdown`

TODO: Make a version that prints normally (like display(df)) + make these little
      headers more informative/useful.

TODO: Make a version (or just add optional parameters to toggle this) where we
      also add the sub-components of WER (substitutions, insertions, deletions).
      This may require re-calculating some stuff, but I actually think that the
      'wer_df' should come with hose columns included...
"""
import pandas as pd

from scipy.stats import pearsonr, mannwhitneyu

# From this project
from ...config import SOURCE_NAMES


# --------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------
def _get_significance_mark(p: float) -> str:
    """Returns a LaTeX superscript asterisk string for significance thresholds."""
    if p < 0.001: return "^{***}"
    if p < 0.01:  return "^{**}"
    if p < 0.05:  return "^{*}"
    return ""
 
def _format_cell(mean: float, std: float) -> str:
    """Formats a mean ± SD string for LaTeX."""
    return f"{mean:.1f} $\\pm$ {std:.1f}"

# ================================================================================
# Generate the WER table in LaTeX format
# ================================================================================
def generate_wer_latex_table(
    res_base      : pd.DataFrame,  # Combined results DataFrame (contains 'dx', 'mmse', 'age', 'srcID', 'WER')
    wer_df        : pd.DataFrame,  # WER DataFrame (fallback; res_base is preferred since it has metadata) 
    target_sources: list = None,   # List of srcID values to include (in order)
) -> None:
    """
    Prints LaTeX table rows showing per-group WER stats, delta WER (AD - CN),
    and Pearson correlation of WER with MMSE for each ASR source.
 
    WER values are multiplied by 100 to express as percentages.
    """
    if target_sources is None: target_sources = SOURCE_NAMES
        
    # Working copy of the results DataFrame
    wer_dx = res_base.copy()

    # Print a header
    print("\n" + "=" * 30)
    print("LATEX TABLE ROWS (Copy/Paste)")
    print("=" * 30)
 
    # --------------------------------------------------------------------------------
    # Generate a row for each source
    # --------------------------------------------------------------------------------
    for src in target_sources:
        # Get data for this specific source
        subset: pd.DataFrame = wer_dx[wer_dx["srcID"] == src]
 
        if subset.empty:
            print(f"{src} & N/A & N/A & N/A & N/A \\\\")
            continue
 
        # Split by group; multiply by 100 for percentage display
        ctrl = subset[subset["dx"] == "Control"   ]["WER"] * 100
        ad   = subset[subset["dx"] == "ProbableAD"]["WER"] * 100
 
        wer_ctrl_str = _format_cell(ctrl.mean(), ctrl.std())
        wer_ad_str   = _format_cell(ad  .mean(), ad  .std())
 
        # --------------------------------------------------------------------------------
        # Delta WER (AD - CN) & Mann-Whitney significance
        # --------------------------------------------------------------------------------
        delta = ad.mean() - ctrl.mean()
        try:
            _, p_mw = mannwhitneyu(ctrl, ad)
            sig_mw  = _get_significance_mark(p_mw)
        except Exception:
            sig_mw = ""
            
        delta_str = f"{delta:.1f}{sig_mw}"
 
        # --------------------------------------------------------------------------------
        # Pearson correlation of WER vs MMSE
        # --------------------------------------------------------------------------------
        valid_corr = subset.dropna(subset=["WER", "mmse"])
        if len(valid_corr) > 1:
            r, p_corr = pearsonr(valid_corr["WER"], valid_corr["mmse"])
            corr_str  = f"{r:.2f}{_get_significance_mark(p_corr)}"
        else:
            corr_str = "NaN"
 
        # Print the full row
        print(f"{src:<20} & {wer_ctrl_str:>16} & {wer_ad_str:>16} & {delta_str:>11} & {corr_str:>11} \\\\")
 
    # Print tail footer
    print("=" * 30)
    print("Note: * p<0.05, ** p<0.01, *** p<0.001")
    print("WER values multiplied by 100 to show as percentages")
 
