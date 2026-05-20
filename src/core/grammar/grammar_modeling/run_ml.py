"""
Run ML models to get OOF biomarker scores.
--------------------------------------------------------------------------------
`src.core.grammar.grammar_modeling.run_ml`

"""
import numpy  as np
import pandas as pd

# Model types
from sklearn.linear_model  import LinearRegression, LogisticRegression, Lasso

# From this project
from ....utils.logging.logging import RESET, BOLD, UNBOLD, BLUE
   
from  .prepare_features import prep_ml_inputs
from  .ml_models     import linear_regression_oof, logistic_classifier_oof


# ================================================================================
# Use ML models to get OOF scores for biomarkers
# ================================================================================
def run_ml_for_biomarker(
    feats         : pd.DataFrame,
    *,
    verbose       : int  = 1,
    clip          : bool = False,
    clip_val      : int  = 25,
    source        : str  = "Source",
):
    # Prepare features
    _, X, y_reg, y_cls = prep_ml_inputs(feats, verbose=verbose, clip=clip, clip_val=clip_val)
    if verbose == 1: print(f"\n{'-'*75}\nRunning ML Models for {source}\n{'-'*75}")

    # 1) Run regression model
    if verbose == 1: print(f"\n{BOLD}Regression on MMSE:{RESET}             {BLUE}{BOLD}Metrics using all OOF preds:{RESET}")

    lasso_params = dict(model_cls=Lasso, reg_params={"alpha":1.0, "fit_intercept":True})
    oof_reg, summary_reg = linear_regression_oof(X, y_reg, verbose=verbose)# **lasso_params)

    # 2) Run classification model
    if verbose == 1: print(f"\n{BOLD}Classification on dx:{RESET}           {BLUE}{BOLD}Metrics using all OOF preds:{RESET}")
  
    logreg_params={'C': 1.0, 'class_weight': 'balanced',  "l1_ratio": 1, 'solver': 'saga'}
    oof_clf, summary_clf = logistic_classifier_oof(X=X, y=y_cls, logreg_params=logreg_params, verbose=verbose)
    
    # Format Results
    res = feats[[ "pID", "age", "sex", "dx", "mmse"]].copy() # "srcID",
    res = res[res["mmse"].notna()]
    res["reg"] = oof_reg["oof_pred"]
    res["cls"] = oof_clf["oof_proba_pos"]

    return res, dict(reg=summary_reg, cls=summary_clf)

