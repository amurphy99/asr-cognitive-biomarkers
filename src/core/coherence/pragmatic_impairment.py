"""
Calculate the Pragmatic Impairment biomarker.
--------------------------------------------------------------------------------
`src.core.coherence.PragmaticImpairment`

This is pretty much just "cosine similarity of each sentence (utterance) with 
the rest of the text." Some other methods use this as just one of multiple other
features, similar to Altered Grammar, but this one seems like it is highly
explainable (e.g. "you were repetitive"), so it gets its own distinction as a 
biomarker. 

Used to use sklearn for cosine similarity, but now we have our own 
implementation that is faster for this use case.
    `from sklearn.metrics.pairwise import cosine_similarity`


TODO: Might eventually need to have a way to "go backwards" with this and trace
      a poor biomarker score to the words that caused it...

      
"""
import numpy  as np
import pandas as pd

# From this project
from .utils.general      import cosine_similarity
from .utils.validate     import validation_setup
from .utils.prepare_text import process_input_data

# Pre-defined models
from .models.coherence_models import load_models
EMBEDDING_VECTORS, ENTROPY, STOP_LIST = load_models()

# --------------------------------------------------------------------------------
# Builds a weighted mean embedding for a token window
# --------------------------------------------------------------------------------
def _weighted_mean_embed(
    ids           : np.ndarray,    # Integer row indices into vecs (0..V-1)
    weights       : np.ndarray,    # Base weights (freq/logfreq/ones) per unique id
    *,
    vecs          : np.ndarray,    # (V, D) embedding matrix (vectors.values)
    entropy_arr   : np.ndarray,    # (V,) normalized entropy values in ~[0,1], aligned with vecs
    entropy_alpha : float = 0.5,   # How strongly to weight by entropy (0 = ignore entropy; 1 = full)
    normalize     : bool  = True,  # L2 normalize each token vector before averaging
) -> np.ndarray:
    """Returns vector of shape (D,). If no valid tokens or sum of weights is zero, returns zeros."""
    if ids.size == 0: return np.zeros((vecs.shape[1],), dtype=np.float32)

    # Entropy weighting: stabilised and bounded
    entropy_w = entropy_arr[ids].astype(np.float32)
    entropy_w = np.clip(entropy_w, 0.0, 1.0)  # just in case

    # Convex combination: (1 - alpha) + alpha * entropy_w
    entropy_factor = (1.0 - entropy_alpha) + entropy_alpha * entropy_w

    # Apply entropy weights
    weights = weights.astype(np.float32) * entropy_factor
    M       = vecs[ids, :].astype(np.float32, copy=False)

    # L2 normalize
    if normalize:
        d         = np.linalg.norm(M, axis=1)
        d[d == 0] = 1.0
        M         = M / d[:, None]

    # Return 0 if sum of weights is 0
    s = weights.sum()
    if s == 0: return np.zeros((vecs.shape[1],), dtype=np.float32)

    return (weights[:, None] * M).sum(axis=0) / s

# --------------------------------------------------------------------------------
# Weigh the window based on the token frequencies
# --------------------------------------------------------------------------------
def _weigh_window_embedding(*, counts: np.ndarray, method: str = "none") -> np.ndarray:
    if   method == "logfreq" : return np.log10(counts.astype(np.float32) + 1.0)
    elif method == "freq"    : return counts.astype(np.float32)
    elif method == "none"    : return np.ones_like(counts, dtype=np.float32)
    else: raise ValueError(f"Bad 'f_weight' argument; given: {method}")

# --------------------------------------------------------------------------------
# Cosine similarity each window with the mean of all other windows
# --------------------------------------------------------------------------------
def _local_coherence(*, window_embeddings: np.ndarray, valid_windows: np.ndarray) -> np.ndarray:
    """
    We get the mean of all windows first, so in the loop we can remove the i-th 
    window from the overall mean to get the mean of all other windows rather 
    than recalculating the mean of other windows for every iteration. 
    """
    # 0) Mean embedding of ALL windows
    mean_all = window_embeddings[valid_windows].mean(axis=0)
    idxs     = np.where(valid_windows)[0]
    n_valid  = len(idxs)

    # Cosine similarity of EACH window with the mean of ALL OTHER windows
    window_coherence = []
    for j in idxs:
        # 1) Remove this window's embedding from the overall mean embedding
        mean_others = (mean_all * n_valid - window_embeddings[j, :]) / (n_valid - 1)

        # 2) Calculate the cosine similarity of this window with the mean of the other windows
        window_coherence.append(cosine_similarity(window_embeddings[j, :], mean_others))

    # 3) Return the float array
    window_coherence = np.asarray(window_coherence, dtype=np.float32)
    return window_coherence

# --------------------------------------------------------------------------------
# Aggregation: mean / p10 / trimmed mean
# --------------------------------------------------------------------------------
def _aggregate_cos(*, cos_arr: np.array, method: str, trim: float = 0.10) -> float:
    """
    Aggregates cosine scores using different methods:
        * "mean"   : simple mean of all window cosines
        * "p10"    : 10th percentile (focus on weaker-coherence regions)
        * "trimmed": mean after trimming `trim` fraction from each tail
    """
    if   method == "mean"   : return float(cos_arr.mean())
    elif method == "p10"    : return float(np.quantile(cos_arr, 0.10))
    elif method == "trimmed":
        # If invalid trim or too few points, fall back to mean
        frac = float(trim)
        if (frac <= 0.0) or (frac >= 0.5) or (cos_arr.size < 3): 
            return float(cos_arr.mean())

        # Mask out the edge percentiles
        lower_q = frac
        upper_q = 1.0 - frac
        q_low, q_high = np.quantile(cos_arr, [lower_q, upper_q])
        mask = (cos_arr >= q_low) & (cos_arr <= q_high)

        if mask.sum() == 0: return float(cos_arr.mean())
        else:               return float(cos_arr[mask].mean())

    else: raise ValueError(f"'agg' must be one of: 'mean', 'p10', 'trimmed' | Got: {method}")


# ================================================================================
# Pragmatic Global Coherence (user text ONLY)
# ================================================================================
def pragmatic_global_coherence(
    data      : pd.DataFrame,  # DataFrame with all user text; requires columns: ["participants", "response"]
    *,
    # Models (loaded from saved files)
    embedding_vectors  : pd.DataFrame = EMBEDDING_VECTORS,  # Token embedding vectors. rows: tokens, cols: dims
    entropy            : pd.DataFrame = ENTROPY,            # Pre-defined values. length == vectors.shape[0], col 'x', aligned by row order
    stop_list          : pd.DataFrame = STOP_LIST,          # One-column DataFrame of stopwords
    
    # Windowing config
    window_size           : int   = 10,          # Token window size
    min_tokens_per_window : int   =  3,          # Force minimum content per window

    # Coherence config
    frequency_weighting   : str   = "logfreq",   # How to weigh tokens based on their frequencies: "logfreq" | "freq" | "none"
    aggregation_method    : str   = "mean",      # How to aggregate cosine scores: "mean" | "p10" | "trimmed"
    trim                  : float = 0.10,        # Fraction to trim for "trimmed" mean
    entropy_alpha         : float = 0.5,         # Strength of entropy weighting
    normalize             : bool  = True,        # L2 normalize each token vector before averaging
) -> float:
    """
    Pragmatic (semantic) global coherence biomarker.
    - Concatenates all USER turns into one long text.
    - Builds sliding-window embeddings around each token.
    - For each valid window, compares its embedding to the mean of all OTHER window embeddings (internal coherence).

    Returns a single float "global coherence" value (0.0 if not enough valid content).
    """
    # Setup vectors, entropy, mapping (D == embedding size)
    vecs, entropy_arr, D, tok2id = validation_setup(
        frequency_weighting=frequency_weighting, window_size=window_size, embedding_vectors=embedding_vectors, entropy=entropy)
    weighted_mean_args = dict(normalize=normalize, vecs=vecs, entropy_arr=entropy_arr, entropy_alpha=entropy_alpha)

    # Stopwords
    stop_words = stop_list.iloc[:, 0].astype(str).str.lower().tolist()
    stop_set   = set(stop_words)

    # Prepare input data
    _, user_text, _, tokens = process_input_data(data, stop_set=stop_set)
    if (not user_text) or (not tokens): return 0.0

    # --------------------------------------------------------------------------------
    # Window Embeddings
    # --------------------------------------------------------------------------------
    W       = len(tokens)
    win_emb = np.zeros((W, D), dtype=np.float32)

    # Build window embeddings
    for i in range(W):
        a = max(0, i - window_size + 1)
        b = i + 1  # inclusive of i
        span_tokens = [t for t in tokens[a:b] if t in tok2id]

        # Require a minimum number of known tokens per window
        if (len(span_tokens) < min_tokens_per_window): continue

        ids = np.array([tok2id[t] for t in span_tokens], dtype=np.int64)
        uniq, counts = np.unique(ids, return_counts=True)

        # Apply weighting & get the mean embedding for the window
        weights       = _weigh_window_embedding(counts=counts, method=frequency_weighting)
        win_emb[i, :] = _weighted_mean_embed(uniq, weights, **weighted_mean_args)

    # --------------------------------------------------------------------------------
    # Local & global coherence via cosine similarity
    # --------------------------------------------------------------------------------
    # Only consider windows that actually have a non-zero embedding
    valid = (np.linalg.norm(win_emb, axis=1) > 0)
    if valid.sum() < 2: return 0.0

    # Get the cosine similarity of each window with the mean of all other windows
    window_coherence = _local_coherence(window_embeddings=win_emb, valid_windows=valid)
    if len(window_coherence) == 0: return 0.0

    # Aggregate into a "global" coherence value and return
    global_coherence = _aggregate_cos(cos_arr=window_coherence, method=aggregation_method, trim=trim)
    return global_coherence

