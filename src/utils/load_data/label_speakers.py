"""
Prepare input DataFrame for grammar feature extraction.
--------------------------------------------------------------------------------
`src.utils.load_data.label_speakers`

"""
import numpy  as np
import pandas as pd

import re
from typing import List, Optional, Dict


# --------------------------------------------------------------------------------
# Normalization / token cleanup
# --------------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)
def norm_word(w: str) -> str:
    if w is None: return ""
    w = w.lower().strip()
    w = w.replace("`", "'").replace("`", "'")
    w = _PUNCT_RE.sub("", w)
    return w


# ================================================================================
# Levenshtein alignment (WER) with backtrace
# ================================================================================
# Returns ops: list of (op, truth_i, hyp_i)
# op in {"equal","sub","del","ins"}
def align_words(truth: List[str], hyp: List[str]) -> List[tuple[str, Optional[int], Optional[int]]]:

    # --------------------------------------------------------------------------------
    # Initialize DP
    # --------------------------------------------------------------------------------
    n, m = len(truth), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.empty((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = ("del", i - 1, None)
    
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = ("ins", None, j - 1)

    bt[0, 0] = (None, None, None)

    # --------------------------------------------------------------------------------
    # Fill DP
    # --------------------------------------------------------------------------------
    for i in range(1, n + 1):
        ti = truth[i - 1]
        for j in range(1, m + 1):
            hj       = hyp[j - 1]
            cost_sub = 0 if ti == hj and ti != "" else 1

            # Candidates: (cost, op)
            candidates = [
                (dp[i - 1, j    ] + 1,        ("del", i - 1, None )),
                (dp[i,     j - 1] + 1,        ("ins", None,  j - 1)),
                (dp[i - 1, j - 1] + cost_sub, ("equal" if (cost_sub == 0) else "sub", i - 1, j - 1)),
            ]

            # Tie-breaker prefers "equal" > "sub" > "del" > "ins" when costs equal
            def rank(op_tuple):
                op = op_tuple[0]
                return {"equal": 0, "sub": 1, "del": 2, "ins": 3}.get(op, 9)

            best_cost = min(c[0] for c in candidates)
            best_ops  = [c for c in candidates if c[0] == best_cost]
            best      = sorted(best_ops, key=lambda x: rank(x[1]))[0]

            dp[i, j] = best[0]
            bt[i, j] = best[1]

    # --------------------------------------------------------------------------------
    # Backtrace
    # --------------------------------------------------------------------------------
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op, ti, hj = bt[i, j]
        ops.append((op, ti, hj))
        if   op in ("equal", "sub"): i -= 1; j -= 1
        elif op == "del"           : i -= 1
        elif op == "ins"           : j -= 1
        else: break

    ops.reverse()
    return ops


# ================================================================================
# Timestamp fallback for insertions
# ================================================================================
def pick_speaker_by_time(
    hand_group  : pd.DataFrame,
    s           : float,
    e           : float,
    speaker_col : str = "speaker",
    start_col   : str = "start",
    end_col     : str = "end",
) -> Optional[str]:
    """ 
    Choose speaker with max overlap with [s,e]. If no overlap, choose nearest midpoint.
    """
    if hand_group.empty: return None

    s   = float(s)
    e   = float(e)
    mid = (s + e) / 2.0

    hs = pd.to_numeric(hand_group[start_col], errors="coerce").to_numpy()
    he = pd.to_numeric(hand_group[  end_col], errors="coerce").to_numpy()

    valid = np.isfinite(hs) & np.isfinite(he) & (he >= hs)
    if not valid.any(): return None

    hs_v = hs[valid]
    he_v = he[valid]
    sp_v = hand_group.loc[valid, speaker_col].astype(str).to_numpy()

    # Overlap
    overlap = np.maximum(0.0, np.minimum(he_v, e) - np.maximum(hs_v, s))
    if overlap.max() > 0:
        idx = int(overlap.argmax())
        return str(sp_v[idx])

    # Nearest midpoint
    hmid = (hs_v + he_v) / 2.0
    idx = int(np.argmin(np.abs(hmid - mid)))
    return str(sp_v[idx])


# ================================================================================
# Label ASR words with hand speakers via Levenshtein alignment
# ================================================================================
def label_asr_speakers(
    ref_df          : pd.DataFrame,
    asr_df          : pd.DataFrame,
    
    # If ASR doesn't have an uttID, set group_col=None OR add it before calling
    group_col        : Optional[str] = None,   # None for global alignment
    
    ref_word_col    : str = "word",
    asr_word_col    : str = "word",
    
    ref_speaker_col : str = "pID",
    
    ref_start_col   : str = "s",
    asr_start_col   : str = "s",
    ref_end_col     : str = "e",
    asr_end_col     : str = "e",
    
) -> pd.DataFrame:
    # --------------------------------------------------------------------------------
    # Set up comparison DataFrames
    # --------------------------------------------------------------------------------
    ref_df = fill_hand_times_from_utt_bounds(ref_df.copy())

    ref = ref_df.copy()
    asr = asr_df.copy()

    ref["_w"] = ref[ref_word_col].astype(str).map(norm_word)
    asr["_w"] = asr[asr_word_col].astype(str).map(norm_word)

    if group_col is None:
        ref["_grp"] = "__all__"
        asr["_grp"] = "__all__"
        group_col = "_grp"
    else:
        if group_col not in ref.columns: raise ValueError(f"ref_df missing group_col={group_col}")
        if group_col not in asr.columns: raise ValueError(f"asr_df missing group_col={group_col}. Set group_col=None or add the column.")

    out_rows = []

    # --------------------------------------------------------------------------------
    # Compare words
    # --------------------------------------------------------------------------------
    for grp, ref_g in ref.groupby(group_col, sort=False):
        asr_g = asr[asr[group_col] == grp]
        if asr_g.empty: continue

        truth_words = ref_g["_w"].tolist()
        hyp_words   = asr_g["_w"].tolist()

        ops = align_words(truth_words, hyp_words)

        # Build a mapping hyp_index -> truth_index (or None)
        hyp_to_truth: Dict[int, Optional[int]] = {j: None       for j in range(len(hyp_words))}
        hyp_op      : Dict[int, str          ] = {j: "unmapped" for j in range(len(hyp_words))}

        for op, ti, hj in ops:
            if hj is None: continue
            if op in ("equal", "sub"):
                hyp_to_truth[hj] = ti
                hyp_op[hj] = op

            elif op == "ins":
                hyp_to_truth[hj] = None
                hyp_op[hj] = "ins"

        # Pull truth speaker per truth index
        truth_speakers = ref_g[ref_speaker_col].astype(str).tolist()
        truth_orig_w   = ref_g[ref_word_col   ].astype(str).tolist()

        # Attach results to asr_g in original order
        asr_g   = asr_g.reset_index(drop=False)  # keep original index for stable merge back
        labeled = asr_g.copy()

        speaker_true       = []
        aligned_truth_word = []
        aligned_truth_idx  = []
        aligned_op         = []

        for j in range(len(labeled)):
            ti = hyp_to_truth.get(j, None      )
            op = hyp_op      .get(j, "unmapped")

            if ti is not None:
                speaker_true      .append(truth_speakers[ti])
                aligned_truth_word.append(truth_orig_w  [ti])
                aligned_truth_idx .append(int(ti))
                aligned_op        .append(op)
            
            # If the operation was insertion / unmapped -> use the timestamp fallback
            else:
                s = labeled.iloc[j][asr_start_col]
                e = labeled.iloc[j][asr_end_col  ]
                sp = pick_speaker_by_time(ref_g, s, e, speaker_col=ref_speaker_col, start_col=ref_start_col, end_col=ref_end_col)
                
                speaker_true      .append(sp  )
                aligned_truth_word.append(None)
                aligned_truth_idx .append(None)
                aligned_op        .append(op  )

        # Book-keeping
        labeled["speaker_true"  ] = speaker_true
        labeled["align_op"      ] = aligned_op
        labeled["truth_word"    ] = aligned_truth_word
        labeled["truth_word_idx"] = aligned_truth_idx

        out_rows.append(labeled)

    if not out_rows:
        return asr_df.assign(speaker_true=None, align_op=None, truth_word=None, truth_word_idx=None)

    labeled_all = pd.concat(out_rows, ignore_index=True)

    # Restore to original ASR row order (if we preserved index)
    if "index" in labeled_all.columns:
        labeled_all = labeled_all.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    return labeled_all


# --------------------------------------------------------------------------------
# Try to fill in start/end timestamps too (if possible)
# --------------------------------------------------------------------------------
def fill_hand_times_from_utt_bounds(
    hand_df   : pd.DataFrame,
    group_col : str = "uttID",
    start_col : str = "s",
    end_col   : str = "e",
) -> pd.DataFrame:
    """
    If an utterance has *some* start/end times (often only on first/last word),
    fill missing word-level start/end by evenly spacing between utterance bounds.
    """
    out = hand_df.copy()
    out[start_col] = pd.to_numeric(out[start_col], errors="coerce")
    out[end_col]   = pd.to_numeric(out[end_col],   errors="coerce")

    # preserve original order
    out["_orig_i"] = np.arange(len(out))

    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("_orig_i").copy()
        n = len(g)
        if n == 0:
            return g

        # utterance bounds from whatever is present
        u_start = g[start_col].min(skipna=True)
        u_end   = g[end_col].max(skipna=True)

        if pd.isna(u_start) or pd.isna(u_end) or not np.isfinite(u_start) or not np.isfinite(u_end):
            return g
        if u_end <= u_start:
            return g

        edges = np.linspace(float(u_start), float(u_end), n + 1)
        fill_starts = edges[:-1]
        fill_ends   = edges[1:]

        # Fill missing only
        start_nan = g[start_col].isna().to_numpy()
        end_nan   = g[end_col].isna().to_numpy()

        g.loc[start_nan, start_col] = fill_starts[start_nan]
        g.loc[end_nan,   end_col]   = fill_ends[end_nan]

        return g

    out = out.groupby(group_col, group_keys=False, sort=False).apply(_fill_group, include_groups=False)
    out = out.sort_values("_orig_i").drop(columns=["_orig_i"]).reset_index(drop=True)
    return out

