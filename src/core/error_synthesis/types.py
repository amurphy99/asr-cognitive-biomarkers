"""
Type aliases and dataclasses for the ASR corruption module.
--------------------------------------------------------------------------------
`src.core.error_synthesis.types`

"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing      import List, Tuple, Dict, Optional


# --------------------------------------------------------------------------------
# Type Aliases
# --------------------------------------------------------------------------------
Token       = str
Sentence    = List[Token]
AlignmentOp = Tuple[str, Optional[str], Optional[str]]  # (op_type, ref_token, hyp_token)

# --------------------------------------------------------------------------------
# Counts of edit operations from a single alignment pass
# --------------------------------------------------------------------------------
@dataclass
class EditStats:
    hits : int = 0   # exact matches
    subs : int = 0   # substitutions
    dels : int = 0   # deletions
    ins  : int = 0   # insertions

    @property
    def wer(self) -> float:
        """
        Word Error Rate = (S + D + I) / N
        """
        ref_len = self.hits + self.subs + self.dels
        if ref_len == 0: return 0.0 if (self.ins == 0) else 1.0
        else:            return (self.subs + self.dels + self.ins) / ref_len


# --------------------------------------------------------------------------------
# ErrorProfile
# --------------------------------------------------------------------------------
@dataclass
class ErrorProfile:
    """
    Learned error distributions for the corruption engine (confusion matrix, 
    ins/del counts, dropout stats).
    """
    # Confusion matrix: ref_word -> Counter({hyp_word: count})
    sub_map : Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    # Global counts for fallback distributions
    ins_counter : Counter = field(default_factory=Counter)
    del_counter : Counter = field(default_factory=Counter)
    hyp_vocab   : Counter = field(default_factory=Counter)

    # Operation probabilities (sub vs del vs ins)
    op_counts : Counter = field(default_factory=Counter)

    # Dropout statistics (lengths of continuous missing segments)
    dropout_lengths   : List[int] = field(default_factory=list)
    dropout_threshold : float     = 0.05  # ratio of hyp/ref length to consider a segment "dropped"

