"""
Word Error Rate (WER)
--------------------------------------------------------------------------------
`src.utils.analysis.word_error_rate`

"""
import pandas as pd

import jiwer
from typing import Dict, List


# ================================================================================
# Calculate WER
# ================================================================================
def calculate_wer_per_pid(
    gt_dict      : Dict[str, List[str]],
    system_dicts : Dict[str, Dict[str, List[str]]],
    verbose = 1
) -> pd.DataFrame:
    """
    Calculate WER per pID, per dataset.

    gt_dict:       {pID: [sent1, sent2, ...]}  (ground truth)
    system_dicts:  {"whisper_large": {pID: [...], ...},
                    "whisper_tiny":  {pID: [...], ...},
                    "google":        {pID: [...], ...}, ...}

    Returns a DataFrame with columns:
        ["pID", "dataset", "wer", "substitutions", "deletions", "insertions", "reference_length"]
    """
    all_pids = set(gt_dict.keys())
    if verbose: print(f"Ground-truth pIDs: {len(all_pids)}")
        
    rows = []

    # Only use pIDs that exist in the ground truth
    all_pids = set(gt_dict.keys())

    for dataset_name, sys_dict in system_dicts.items():
        # intersect pIDs in gt and this system
        common_pids = all_pids.intersection(sys_dict.keys())

        for pid in common_pids:
            ref_text = _join_sentences( gt_dict[pid])
            hyp_text = _join_sentences(sys_dict[pid])
            
            ref_words = len(ref_text.split(" "))
            hyp_words = len(hyp_text.split(" "))
            
            out = jiwer.process_words(ref_text, hyp_text)

            rows.append(
                {
                    "pID"   : pid,
                    "srcID" : dataset_name,
                    
                    "WER": out.wer, 
                    "MER": out.mer,
                    "WIL": out.wil,
                    "WIP": out.wip,
                    
                    "sub_r": out.substitutions / ref_words,
                    "ins_r": out.insertions    / ref_words,
                    "del_r": out.deletions     / ref_words,
                }
            )

    return pd.DataFrame(rows)


# Can take either a list of sentences or a single string
def _join_sentences(x):
    if isinstance(x, list): return " ".join(x)
    return str(x)

