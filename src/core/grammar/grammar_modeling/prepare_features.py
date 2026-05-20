"""
Load transcript lists for feature generation.
--------------------------------------------------------------------------------
`src.core.grammar.grammar_modeling.prepare_features`

"""


# ================================================================================
# Finish preparing the data
# ================================================================================
def prep_ml_inputs(feats, verbose=1, clip=False, clip_val=25, drop_features=[], group_col=None):
    """
    Meta columns: 'age', 'sex', 'edu', 'lang', 'dx', 'mmse', 'MoCA'
    """
    # Parameters
    USE_AGE  = False # True False
    USE_SEX  = False
    RAND_ST  = 0
    N_SPLITS = 10 
    if verbose == 1: print(f"USE_AGE={USE_AGE}, USE_SEX={USE_SEX}, RAND_ST={RAND_ST}, N_SPLITS={N_SPLITS}")

    # Working copy of the data
    d4 = feats.copy(); len1 = len(d4)

    # Drop columns where MMSE is empty
    d4 = d4[d4["mmse"].notna()]
    if clip:
        #d4 = d4[d4["mmse"] >= 20 ]
        d4["mmse"] = d4["mmse"].clip(lower=clip_val) # 20 25
    
    if verbose == 1: print(f"Orig len: {len1}, len after: {len(d4)}\n")
  
    # Prepare X data
    drop_cols = ["pID", 'dx', 'mmse', "pittID", 'MoCA', 'edu', 'lang', "name"]
    if not USE_AGE: drop_cols.append("age")
    if not USE_SEX: drop_cols.append("sex")
    if group_col is None: drop_cols.append("srcID")

    # Remove features based on input args
    for col in drop_features:
        if col not in drop_cols: drop_cols.append(col)

    x = d4.copy()
    for col in drop_cols:
        if col in feats.columns: x.drop([col], axis=1, inplace=True)

    # Prepare y data for both types
    y_reg = d4["mmse"].copy()
    #y_cls = (d4["dx"] == "Control").astype(int)
    y_cls = (d4["dx"] == "ProbableAD").astype(int)

    return d4, x, y_reg, y_cls

