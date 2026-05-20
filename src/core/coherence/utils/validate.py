"""
Validate input configuration for Pragmatic Impairment.
--------------------------------------------------------------------------------
`src.core.coherence.utils.validate`

Some of the "setup" stuff here seems like it should maybe just be done once, but
if it runs quickly, maybe it is better to not keep it in memory...

TODO: Should definitely redo this for the entropy thing we load in...
      We should just do everything we can that isn't related to the input
      arguments and then save new copies of the files.

"""
import numpy  as np
import pandas as pd


# ================================================================================
# Model Validation / Setup
# ================================================================================
def validation_setup(
    frequency_weighting : str,           # How to weigh tokens based on their frequencies: "logfreq" | "freq" | "none"
    window_size         : int,           # Token window size
    embedding_vectors   : pd.DataFrame,  # Token embedding vectors. rows: tokens, cols: dims
    entropy             : pd.DataFrame,  # Pre-defined values. length == vectors.shape[0], col 'x', aligned by row order
):
    """
    Validate inputs and return:
      vecs        : (V, D) embedding matrix (float32)
      entropy_arr : (V,) entropy array (float32)
      D           : embedding dimension
      tok2id      : dict mapping token -> row index
    """
    # --------------------------------------------------------------------------------
    # Validation Step 1: input configs
    # --------------------------------------------------------------------------------
    if frequency_weighting not in {"logfreq", "freq", "none"} : raise ValueError(f"f_weight must be one of {'logfreq','freq','none'}; got: {frequency_weighting}")
    if window_size < 1                                        : raise ValueError(f"window_size must be >= 1; got: {window_size}")
    if not isinstance(embedding_vectors, pd.DataFrame)        : raise ValueError(f"vectors must be a DataFrame with token index; got: {type(embedding_vectors)}")

    # Token -> row index mapping (vectors.index are words)
    V, D   = embedding_vectors.shape
    vecs   = embedding_vectors.values.astype(np.float32, copy=False)
    tok2id = {w: i for i, w in enumerate(embedding_vectors.index)}

    # Entropy must be a DataFrame with column 'x', same length as vectors
    if not isinstance(entropy, pd.DataFrame) : raise ValueError("entropy must be a DataFrame")
    if "x" not in entropy.columns            : raise ValueError("entropy DataFrame must have a column named 'x'")
    if len(entropy) != V                     : raise ValueError("entropy length must equal number of vector rows")

    # --------------------------------------------------------------------------------
    # Stabilize entropy: z-score then sigmoid -> (0,1)
    # --------------------------------------------------------------------------------
    e = entropy["x"].to_numpy(dtype=np.float32)

    # Ignore entropy if not all values are finite OR variance is too low
    if not (np.all(np.isfinite(e))) or (e.std() < 1e-8): 
        entropy_arr = np.ones_like(e, dtype=np.float32)

    # Stabilize entropy values
    else:
        # Z-score
        entropy_mean = float(e.mean())
        entropy_std  = float(e.std ())
        entropy_z    = (e - entropy_mean) / (entropy_std + 1e-8)

        # Sigmoid ("logistic squashing")
        entropy_arr = 1.0 / (1.0 + np.exp(-entropy_z))

    # Return setup
    return vecs, entropy_arr, D, tok2id

