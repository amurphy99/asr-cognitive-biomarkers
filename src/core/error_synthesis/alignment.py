"""
Levenshtein alignment between reference and hypothesis token sequences.
--------------------------------------------------------------------------------
`src.core.error_synthesis.alignment`

"""
from typing import List, Tuple

# From this project
from .types import Sentence, AlignmentOp, EditStats


# ================================================================================
# Handle aligning reference and hypothesis text
# ================================================================================
class AlignmentEngine:

    @staticmethod
    def align(ref: Sentence, hyp: Sentence) -> Tuple[List[AlignmentOp], EditStats]:
        """
        Levenshtein alignment; returns (ops, stats) where ops is the sequence of 
        (op, ref_tok, hyp_tok) transforms.
        """
        n, m = len(ref), len(hyp)

        # Dynamic programming matrix: dp[i][j] = min edit distance between ref[:i] and hyp[:j]
        dp  = [[0]    * (m + 1) for _ in range(n + 1)]
        ptr = [[None] * (m + 1) for _ in range(n + 1)]  # backtrace: (op_name, prev_i, prev_j)

        for i in range(1, n + 1): dp[i][0] = i; ptr[i][0] = ("del", i - 1, 0)
        for j in range(1, m + 1): dp[0][j] = j; ptr[0][j] = ("ins", 0, j - 1)

        # --------------------------------------------------------------------------------
        # Fill the DP matrix
        # --------------------------------------------------------------------------------
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost_hit = dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
                cost_del = dp[i - 1][j    ] + 1
                cost_ins = dp[i    ][j - 1] + 1

                # Priority: hit/sub > del > ins (standard convention for stability)
                if cost_hit <= cost_del and cost_hit <= cost_ins:
                    dp [i][j] = cost_hit
                    op        = "hit" if ref[i - 1] == hyp[j - 1] else "sub"
                    ptr[i][j] = (op, i - 1, j - 1)
                elif cost_del <= cost_ins:
                    dp [i][j] = cost_del
                    ptr[i][j] = ("del", i - 1, j)
                else:
                    dp [i][j] = cost_ins
                    ptr[i][j] = ("ins", i, j - 1)

        # --------------------------------------------------------------------------------
        # Backtrace
        # --------------------------------------------------------------------------------
        ops   = []
        stats = EditStats()
        i, j  = n, m

        while i > 0 or j > 0:
            op, pi, pj = ptr[i][j]
            if   op == "hit": ops.append(("hit", ref[i - 1], hyp[j - 1])); stats.hits += 1
            elif op == "sub": ops.append(("sub", ref[i - 1], hyp[j - 1])); stats.subs += 1
            elif op == "del": ops.append(("del", ref[i - 1],       None)); stats.dels += 1
            elif op == "ins": ops.append(("ins",       None, hyp[j - 1])); stats.ins  += 1

            i, j = pi, pj

        ops.reverse()
        return ops, stats

