"""
Run ML models to get OOF biomarker scores.
--------------------------------------------------------------------------------
`src.core.grammar.grammar_modeling.ml_models`

"""
import numpy  as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression, LogisticRegression, Lasso  # default models

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats     import pearsonr

# From this project
from ....utils.logging.logging import BLUE, BOLD, RESET  


# --------------------------------------------------------------------------------
# Helpers (NOTE: most of these are deprecated in newer versions of this code)
# --------------------------------------------------------------------------------
# Print results
def by_fold_metrics(metrics_dict, oof_metrics=None):
    for key, val in metrics_dict.items():
        val = np.array(val)
        out = f"{(key+':'):6} {val.mean():8.4f} ± {val.std():6.4f}"
        if oof_metrics is not None: out += f"        {BLUE}{(key+':'):6} {oof_metrics[key]:8.4f}{RESET}" 
        print(out)

def _coerce_labels(x, name: str):
    # Handle pandas Series with nullable Int64 and NA cleanly
    s = pd.Series(x)
    if s.isna().any(): raise ValueError(f"{name} contains NA/None at positions: {np.where(s.isna())[0][:10]}")
        
    # Force to plain int 0/1 (avoid object dtype); if labels are strings, encode to ints consistently
    try:              arr = s.astype('int64').to_numpy().ravel()
    except Exception: arr = pd.Categorical(s).codes.astype('int64')
    return arr

# Manual RMSE because this version of sklearn doesn't support it (I guess...)
def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

# Pearson correlation between true and predicted values
def corr_score(y_true, y_pred, return_p: bool = False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # If standard deviation is 0, correlation is undefined
    if (np.std(y_true) == 0) or (np.std(y_pred) == 0): 
        if return_p: return np.nan, np.nan
        else:        return np.nan
    
    # pearsonr returns a tuple: (correlation_coefficient, p_value)
    coef, p = pearsonr(y_true, y_pred)

    # Only return the p value if specified
    if return_p: return float(coef), float(p)
    else:        return float(coef)

def guarded_f1_score(y_true, y_pred, **kwargs):
    try:              return f1_score(y_true, y_pred, **kwargs)
    except Exception: return np.nan
    
def guarded_auc_score(y_true, y_score, **kwargs):
    try:              return roc_auc_score(y_true, y_score, **kwargs)
    except Exception: return np.nan



# ================================================================================
# Regression (on MMSE or MoCA)
# ================================================================================
def linear_regression_oof(
    X,                                  # Training features
    y,                                  # Labels (binary or multiclass)
    n_splits     : int         = 5,     # Number of folds (ignored if groups is not None)
    random_state : int         = 0,     # Used for StratifiedKFold shuffling
    reg_params   : dict | None = None,  # Extra parameters to pass to the model
    groups                     = None,  # List of group IDs or a column name (from X) with group labels for each sample (uses StratifiedKFold(n_splits) if None)
    model_cls = LinearRegression,       # Which regressor type to use
    verbose                    = 1,     # Detail included in any printed outputs
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Generic regression with out-of-fold predictions and metrics.
    model_cls can be LinearRegression, Ridge, Lasso, etc.

    Returns
    -------
    oof_df : DataFrame  => index, y_true, oof_pred, fold
    summary : dict      => {"rmse": ..., "mse": ..., "mae": ..., "r2": ...}
    """
    # --------------------------------------------------------------------------------
    # Working copies
    # --------------------------------------------------------------------------------
    X   = pd.DataFrame(X).copy()
    y   = pd.Series   (y).copy()
    idx = X.index.to_numpy()

    # --------------------------------------------------------------------------------
    # Handle groups (for Leave-One-Group-Out)
    # --------------------------------------------------------------------------------
    group_arr = None
    if groups is not None:
        if isinstance(groups, str):
            if groups not in X.columns: raise ValueError(f"groups='{groups}' is not a column in X")
            group_arr = X[groups].to_numpy()
        else: group_arr = np.asarray(groups)
        if group_arr.shape[0] != len(X): raise ValueError("groups must have the same length as X")

    # --------------------------------------------------------------------------------
    # Regression model hyperparameters
    # --------------------------------------------------------------------------------
    params = dict()
    if reg_params: params.update(reg_params)

    # --------------------------------------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------------------------------------
    if group_arr is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_gen = cv.split(X, y)
    else:
        cv = LeaveOneGroupOut()
        split_gen = cv.split(X, y, groups=group_arr)

    # --------------------------------------------------------------------------------
    # Storage for OOF predictions
    # --------------------------------------------------------------------------------
    oof_fold = np.empty(len(X), dtype=np.int16)
    oof_pred = np.empty(len(X), dtype=np.float32)

    fold_metrics = {"rmse": [], "mse": [], "mae": [], "r2": [], "corr": []}

    # --------------------------------------------------------------------------------
    # Cross-validation loop
    # --------------------------------------------------------------------------------
    for fold, (tr, va) in enumerate(split_gen, start=1):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  model_cls(**params)),
        ])

        pipe.fit(X.iloc[tr], y.iloc[tr])

        preds = pipe.predict(X.iloc[va])
        preds = preds.clip(0, 30) 

        oof_pred[va] = preds
        oof_fold[va] = fold

        fold_y_true = y.iloc[va].to_numpy()
        fold_y_pred = preds

        # --------------------------------------------------------------------------------
        # Evaluation by fold
        # --------------------------------------------------------------------------------
        fold_metrics["rmse"].append(rmse_score         (fold_y_true, fold_y_pred))
        fold_metrics["mse" ].append(mean_squared_error (fold_y_true, fold_y_pred))  # squared=False
        fold_metrics["mae" ].append(mean_absolute_error(fold_y_true, fold_y_pred))
        fold_metrics["r2"  ].append(r2_score           (fold_y_true, fold_y_pred))
        fold_metrics["corr"].append(corr_score         (fold_y_true, fold_y_pred))


    # --------------------------------------------------------------------------------
    # Metrics from OOF
    # --------------------------------------------------------------------------------
    rmse = rmse_score         (y, oof_pred)
    mse  = mean_squared_error (y, oof_pred)  # squared=False
    mae  = mean_absolute_error(y, oof_pred)
    r2   = r2_score           (y, oof_pred)
    corr = corr_score         (y, oof_pred)
    
    summary = {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2, "corr": corr}

    # --------------------------------------------------------------------------------
    # Build OOF table
    # --------------------------------------------------------------------------------
    oof_df = pd.DataFrame({
        "index"    : idx,
        "y_true"   : y.to_numpy(),
        "oof_pred" : oof_pred,
        "fold"     : oof_fold,
    }).set_index("index").sort_index()

    # If you want to print here:
    if verbose == 1: by_fold_metrics(fold_metrics, oof_metrics=summary)

    return oof_df, summary#, fold_metrics




# ================================================================================
# Classification (on dx Control/ProbableAD)
# ================================================================================
def logistic_classifier_oof(
    X,                                   # Training features
    y,                                   # Labels (binary or multiclass)
    n_splits      : int         = 5,     # Number of folds (ignored if groups is not None)
    random_state  : int         = 0,     # Used for StratifiedKFold shuffling
    logreg_params : dict | None = None,  # Extra parameters to pass to sklearn.linear_model.LogisticRegression
    groups                      = None,  # List of group IDs or a column name (from X) with group labels for each sample (uses StratifiedKFold(n_splits) if None)
    verbose                     = 1,     # Detail included in any printed outputs
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Logistic Regression with out-of-fold predictions and metrics.

    Returns
    -------
    oof_df  : DataFrame => Contains index, y_true, oof_pred, fold, and probabilities.
    summary : dict      => Overall metrics: {"acc": ..., "f1": ..., "auc": ...}
    """
    # --------------------------------------------------------------------------------
    # Working copies (save index to rebuild out-of-fold predictions)
    # --------------------------------------------------------------------------------
    X   = pd.DataFrame(X).copy()
    y   = pd.Series   (y).copy()
    idx = X.index.to_numpy()

    # --------------------------------------------------------------------------------
    # Handle groups (for Leave-One-Group-Out)
    # --------------------------------------------------------------------------------
    group_arr = None
    if groups is not None:
        if isinstance(groups, str):
            if groups not in X.columns: raise ValueError(f"groups='{groups}' is not a column in X")
            group_arr = X[groups].to_numpy()
        else: group_arr = np.asarray(groups)
        if group_arr.shape[0] != len(X): raise ValueError("groups must have the same length as X")

    # --------------------------------------------------------------------------------
    # Class labels and basic info
    # --------------------------------------------------------------------------------
    classes   = np.unique(y)  # global, consistent order
    n_classes = len(classes)

    # Positive label for binary classification
    if n_classes == 2: pos_label = classes[-1]
    else:              pos_label = None

    # --------------------------------------------------------------------------------
    # Logistic Regression hyperparameters
    # --------------------------------------------------------------------------------
    params = dict(
        l1_ratio = 0, #penalty  = "l2",
        C        = 1.0,
        solver   = "lbfgs",
        max_iter = 5000,
        #n_jobs   = -1,
        # multi_class="auto" by default
    )
    if logreg_params: params.update(logreg_params)

    # --------------------------------------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------------------------------------
    if group_arr is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_gen = cv.split(X, y)
    else:
        cv = LeaveOneGroupOut()
        split_gen = cv.split(X, y, groups=group_arr)

    # --------------------------------------------------------------------------------
    # Storage for OOF predictions
    # --------------------------------------------------------------------------------
    oof_fold = np.empty(len(X), dtype=np.int16)
    oof_pred_cls = np.empty(len(X), dtype=object)

    if n_classes > 2: oof_proba     = np.zeros((len(X), n_classes), dtype=np.float32)
    else:             oof_proba_pos = np.zeros( len(X),             dtype=np.float32)

    fold_metrics = {"acc": [], "f1": [], "auc": []}

    # --------------------------------------------------------------------------------
    # Cross-validation loop
    # --------------------------------------------------------------------------------
    for fold, (tr, va) in enumerate(split_gen, start=1):
        # Pipeline: impute -> scale -> logistic regression
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  LogisticRegression(**params)),
        ])

        pipe.fit(X.iloc[tr], y.iloc[tr])

        # Predictions
        pred_cls = pipe.predict(X.iloc[va])
        proba    = pipe.predict_proba(X.iloc[va])  # shape (n_va, k)

        # Align predicted probabilities to global 'classes' order
        model_classes = pipe.named_steps["model"].classes_
        col_idx       = [int(np.where(model_classes == c)[0][0]) for c in classes]
        proba         = proba[:, col_idx]

        oof_pred_cls[va] = pred_cls
        if n_classes > 2: oof_proba[va, :] = proba
        else:
            # Store positive-class probability (classes[-1] as "positive")
            pos_col = np.where(classes == classes[-1])[0][0]
            oof_proba_pos[va] = proba[:, pos_col]

        oof_fold[va] = fold

        # --------------------------------------------------------------------------------
        # Fold-level metrics
        # --------------------------------------------------------------------------------
        if n_classes == 2:
            fold_y_true = y.iloc[va].to_numpy()
            fold_y_pred = pred_cls

            fold_metrics["acc"].append(accuracy_score(fold_y_true, fold_y_pred))
            fold_metrics["f1" ].append(f1_score(fold_y_true, fold_y_pred, pos_label=pos_label))

            # AUC for this fold (guarded to handle edge cases)
            fold_y_bin = (fold_y_true == pos_label).astype(int)
            fold_metrics["auc"].append(guarded_auc_score(fold_y_bin, oof_proba_pos[va], average="weighted"))

    # --------------------------------------------------------------------------------
    # Global metrics from OOF predictions
    # --------------------------------------------------------------------------------
    y_lab            = _coerce_labels(y,            "y_true")
    oof_pred_cls_lab = _coerce_labels(oof_pred_cls, "y_pred")

    acc = accuracy_score(y_lab, oof_pred_cls_lab)

    if n_classes == 2:
        f1  = f1_score(y_lab, oof_pred_cls_lab, pos_label=pos_label)
        auc = guarded_auc_score((y_lab == pos_label), oof_proba_pos, average="weighted")
    else:
        f1  = f1_score(y_lab, oof_pred_cls_lab, average="weighted")
        y_onehot = pd.get_dummies(y_lab)
        auc = guarded_f1_score(y_onehot, oof_proba, args=dict(multi_class="ovr", average="weighted"))

    summary = {"acc": acc, "f1": f1, "auc": auc}

    # --------------------------------------------------------------------------------
    # Build OOF table
    # --------------------------------------------------------------------------------
    data = {"index": idx, "y_true": y_lab, "oof_pred": oof_pred_cls_lab, "fold": oof_fold,}

    if n_classes == 2: data["oof_proba_pos"] = oof_proba_pos
    else:
        for j, c in enumerate(classes): data[f"proba_{c}"] = oof_proba[:, j]

    oof_df = pd.DataFrame(data).set_index("index").sort_index()

    # Print per-fold summary vs global OOF summary
    if verbose == 1: by_fold_metrics(fold_metrics, oof_metrics=summary)

    return oof_df, summary

