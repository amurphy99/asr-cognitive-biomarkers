"""
General helpers for calculating coherence / pragmatic impairment.
--------------------------------------------------------------------------------
`src.core.coherence.utils.general`

"""
import numpy as np

# --------------------------------------------------------------------------------
# Cosine Similarity
# --------------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity without sklearn overhead; returns 0.0 if either vector is zero.
    """
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

