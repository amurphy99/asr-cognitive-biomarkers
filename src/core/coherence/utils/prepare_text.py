"""
Prepare input data for coherence analysis.
--------------------------------------------------------------------------------
`src.core.coherence.utils.prepare_text`

"""
import numpy  as np
import pandas as pd

import re

from typing import Tuple

# Config
_TOKEN_RE = re.compile(r"[^\w\s]+")


# --------------------------------------------------------------------------------
# Tokenize sentences into words & remove any stopwords
# --------------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    """
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    return _TOKEN_RE.sub("", text.lower()).strip().split()

def prep_tokens(text: str, stop_set: set[str]) -> list[str]:
    """
    Tokenize and remove stopwords.
    """
    return [t for t in _tokenize(text) if t not in stop_set]


# ================================================================================
# Data Helpers
# ================================================================================
# TODO: This could be moved to somewhere more general & reused for other processing areas
def _find_rename(df: pd.DataFrame, *, target_col: str, col_variations: list[str]) -> pd.DataFrame:
    # If our target column is not in the DataFrame...
    if target_col in df.columns: return df

    # Search for variants of that column name in the DataFrame
    for variant in col_variations:
        # If found, rename to the target and break the loop
        if variant in df.columns: 
            df = df.rename(columns={variant: target_col})
            break

    # Return with the renamed column
    if target_col not in df.columns: raise ValueError(f"Missing {target_col} column in coherence input.")
    return df

# --------------------------------------------------------------------------------
# Initial data formatting
# --------------------------------------------------------------------------------
def process_input_data(data: pd.DataFrame, *, stop_set: set) -> Tuple[pd.DataFrame, list[str], str, list[str]]:
    df = data.copy()

    # Make sure input data has "participants" and "response" columns
    df = _find_rename(df, target_col="participant", col_variations=["role",     ])
    df = _find_rename(df, target_col="response",    col_variations=["full_text",])
    df = df[["participant", "response"]]

    # Collect all USER text and concatenate them together
    user_text = df.loc[df["participant"].str.lower() == "user", "response"].tolist()
    user_text = [x for x in user_text if isinstance(x, str)]

    # Full text
    full_text = " ".join(user_text)
    tokens    = prep_tokens(full_text, stop_set)

    return df, user_text, full_text, tokens

