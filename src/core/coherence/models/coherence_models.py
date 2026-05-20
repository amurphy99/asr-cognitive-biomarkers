"""
Load the models required for the coherence implementation.
--------------------------------------------------------------------------------
`src.core.coherence.models.coherence_models`

"""
import pandas as pd
import os

# Path
COHERENCE_MODELS_PATH = f"{os.path.dirname(os.path.abspath(__file__))}"

# --------------------------------------------------------------------------------
# Load Features for the Pragmatic Score
# --------------------------------------------------------------------------------
def load_models():
    """
    embedding_vectors => Token embedding vectors. rows: tokens, cols: dims
    entropy           => Pre-defined values. length == vectors.shape[0], col 'x', aligned by row order
    stop_list         => One-column DataFrame of stopwords
    """
    embedding_vectors_ = pd.read_csv  (f"{COHERENCE_MODELS_PATH}/new_LSA.csv", index_col=0 )
    entropy_           = pd.read_csv  (f"{COHERENCE_MODELS_PATH}/Hoffman_entropy_53758.csv")
    stop_list_         = pd.read_table(f"{COHERENCE_MODELS_PATH}/stoplist.txt", header=None)
    return embedding_vectors_, entropy_, stop_list_

