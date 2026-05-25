"""
Main class for synthetic ASR error generation.
--------------------------------------------------------------------------------
`src.core.error_synthesis.corruptor`

"""
import random
from collections import Counter
from typing      import List, Dict, Optional, Union, Set

# From this project
from .types      import ErrorProfile
from .text_utils import normalize, tokenize, segment
from .alignment  import AlignmentEngine


# ================================================================================
# Synthesizes ASR-like errors at a target WER
# ================================================================================
class ASRCorruptor:
    def __init__(self, profile: Optional[ErrorProfile] = None):
        self.profile = profile if profile else ErrorProfile()

    # ================================================================================
    # Learn error patterns from real (ref, hyp) pairs
    # ================================================================================
    def fit(self,
        data              : Dict[str, Dict[str, Union[str, List[str]]]],  # { file_id: {'ref': ..., 'hyps': ...} }
        dropout_threshold : float = 0.05,  # hyp/ref length ratio below which a pair is treated as a catastrophic dropout
    ) -> None:
        self.profile.dropout_threshold = dropout_threshold

        for item in data.values():
            # --------------------------------------------------------------------------------
            # Standardize inputs to lists of strings
            # --------------------------------------------------------------------------------
            refs = segment(item["ref"]) if isinstance(item["ref" ],  str) else  item["ref" ]
            hyps =         item["hyps"] if isinstance(item["hyps"], list) else [item["hyps"]]

            # Flatten hypotheses if they are lists of segments
            flat_hyps = []
            for h in hyps:
                if isinstance(h, list): flat_hyps.extend(        h )
                else:                   flat_hyps.extend(segment(h))

            # Normalize and tokenize
            ref_toks_list = [tokenize(normalize(r)) for r in refs     ]
            hyp_toks_list = [tokenize(normalize(h)) for h in flat_hyps]

            # --------------------------------------------------------------------------------
            # Align pairs and update the profile
            # --------------------------------------------------------------------------------
            # Assumes 1:1 mapping or broadcast
            # refs and hyps must be aligned at the segment level when calling
            for i, r_toks in enumerate(ref_toks_list):
                if not r_toks: continue

                h_toks = hyp_toks_list[i] if i < len(hyp_toks_list) else []

                # Catastrophic failure: treat as a dropout, skip the alignment
                if not h_toks or (len(h_toks) / len(r_toks) < dropout_threshold):
                    self.profile.dropout_lengths.append(len(r_toks))
                    continue

                self.profile.hyp_vocab.update(h_toks)

                # Levenshtein alignment
                ops, stats = AlignmentEngine.align(r_toks, h_toks)

                self.profile.op_counts["hit"] += stats.hits
                self.profile.op_counts["sub"] += stats.subs
                self.profile.op_counts["del"] += stats.dels
                self.profile.op_counts["ins"] += stats.ins

                for op, r_w, h_w in ops:
                    if   op == "sub": self.profile.sub_map    [r_w][h_w] += 1
                    elif op == "del": self.profile.del_counter[r_w]      += 1
                    elif op == "ins": self.profile.ins_counter     [h_w] += 1


    # ================================================================================
    # Generate a synthetic hypothesis at the target WER
    # ================================================================================
    def generate(
        self,
        text         : str,
        target_wer   : float,
        seed         : Optional[int] = None,
        dropout_prob : float         = 0.0,
        attempts     : int           = 10,
    ) -> Dict:
        # Setup
        rng      = random.Random(seed)
        ref_toks = tokenize(normalize(text))

        if not ref_toks: return {"text": "", "wer": 0.0, "tokens": []}

        # --------------------------------------------------------------------------------
        # Calculate "error budget"
        # --------------------------------------------------------------------------------
        total_errors      = int(round(target_wer * len(ref_toks)))
        dropout_budget    = int(total_errors * dropout_prob)
        stochastic_budget = max(0, total_errors - dropout_budget)

        # --------------------------------------------------------------------------------
        # Generate (multiple attempts to land as close to target_wer as possible)
        # --------------------------------------------------------------------------------
        best_res  = None
        best_diff = float("inf")

        for _ in range(max(1, attempts)):
            curr_toks = list(ref_toks)

            # 1) Apply dropout (structural deletion)
            self._apply_dropout(rng, curr_toks, dropout_budget)

            # 2) Apply stochastic errors (sub/del/ins)
            self._apply_stochastic(rng, curr_toks, stochastic_budget)

            # 3) Validate result
            _, stats   = AlignmentEngine.align(ref_toks, curr_toks)
            actual_wer = stats.wer
            diff       = abs(actual_wer - target_wer)

            if diff < best_diff:
                best_diff = diff
                best_res  = {"text": " ".join(curr_toks), "wer": actual_wer, "tokens": curr_toks}

            if diff < 0.005: break  # close enough

        return best_res


    # ================================================================================
    # Private helpers
    # ================================================================================
    # Removes connected spans of tokens
    def _apply_dropout(self, rng: random.Random, tokens: List[str], target_drops: int) -> None:
        dropped = 0
        lengths = self.profile.dropout_lengths if self.profile.dropout_lengths else [5, 10, 15]

        while dropped < target_drops and tokens:
            span = rng.choice(lengths)
            span = min(span, target_drops - dropped, len(tokens))
            if span <= 0: break

            start = rng.randint(0, len(tokens) - span)
            del tokens[start : start + span]
            dropped += span

    # Applies individual subs, dels, and ins based on the learned profile
    def _apply_stochastic(self, rng: random.Random, tokens: List[str], budget: int) -> None:
        if budget <= 0 or not tokens: return

        # --------------------------------------------------------------------------------
        # Determine operation mix (default to generic if model is empty)
        # --------------------------------------------------------------------------------
        total = sum(self.profile.op_counts.values()) - self.profile.op_counts["hit"]
        if total > 0:
            p_sub = self.profile.op_counts["sub"] / total
            p_del = self.profile.op_counts["del"] / total
        else:
            p_sub, p_del = 0.6, 0.2  # generic ASR assumption

        n_sub = int(budget * p_sub)
        n_del = int(budget * p_del)
        n_ins = max(0, budget - n_sub - n_del)

        # --------------------------------------------------------------------------------
        # Weighted sampling helper
        # --------------------------------------------------------------------------------
        def pick(counter: Counter, exclude: Optional[Set[str]] = None) -> Optional[str]:
            if not counter: return None
            cands = [w for w in counter if not exclude or w not in exclude]
            if not cands: return None

            weights = [counter[w] for w in cands]
            return rng.choices(cands, weights=weights, k=1)[0]

        # --------------------------------------------------------------------------------
        # Apply edits
        # --------------------------------------------------------------------------------
        # Deletions: prefer words that are commonly deleted
        for _ in range(n_del):
            if not tokens: break
            weights = [self.profile.del_counter[t] + 1 for t in tokens]
            idx     = rng.choices(range(len(tokens)), weights=weights, k=1)[0]
            tokens.pop(idx)

        # Substitutions: prefer specific confusion -> fall back to global hyp vocab
        for _ in range(n_sub):
            if not tokens: break
            idx  = rng.randrange(len(tokens))
            orig = tokens[idx]
            cand = pick(self.profile.sub_map.get(orig)) or pick(self.profile.hyp_vocab)
            if cand: tokens[idx] = cand

        # Insertions
        for _ in range(n_ins):
            idx  = rng.randint(0, len(tokens))
            cand = pick(self.profile.ins_counter) or pick(self.profile.hyp_vocab)
            if cand: tokens.insert(idx, cand)

