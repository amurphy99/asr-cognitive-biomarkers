"""
Helpers for preparing training data and running the corruptor for WER targets.
--------------------------------------------------------------------------------
`src.core.error_synthesis.generate`

"""
import random
from typing    import Dict, List, Callable
from tqdm.auto import tqdm

# From this project
from .corruptor import ASRCorruptor


# ================================================================================
# Prepare/format input data for the ASRCorruptor
# ================================================================================
def build_training_data(
    gt_dict      : Dict[str, List[str]],
    system_dicts : Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, object]]:
    """
    Converts the standard {srcID: {pID: [sent1, ...]}} structure into the format
    expected by ASRCorruptor.fit(): { pID: {"ref": str, "hyps": [str, ...]} }.
    """
    training_data = {}

    for pid, ref_sents in gt_dict.items():
        # Join reference segments into a single transcript string
        ref_text = " ".join(ref_sents) if isinstance(ref_sents, list) else str(ref_sents)

        # Collect hypothesis strings from each ASR system
        hyps_texts = []
        for sys_name, sys_pid_dict in system_dicts.items():
            if pid not in sys_pid_dict: continue

            # Join sentences per pID before passing to the corruptor
            hyp_sents = sys_pid_dict[pid]
            hyp_text  = " ".join(hyp_sents) if isinstance(hyp_sents, list) else str(hyp_sents)
            hyps_texts.append(hyp_text)

        # Only keep pIDs that have at least one hypothesis
        if hyps_texts: 
            training_data[pid] = {"ref": ref_text, "hyps": hyps_texts}

    return training_data


# --------------------------------------------------------------------------------
# Default amount of error that is from "dropout" segments (scaled w target WER)
# --------------------------------------------------------------------------------
def default_dropout_schedule(target_wer: float) -> float:
    if   target_wer <= 0.20: return 0.00
    elif target_wer <= 0.35: return 0.10
    else:                    return 0.20


# ================================================================================
# Generate synthetic transcripts for each target WER, 
# ================================================================================
def generate_synthetic_transcripts(
    corruptor        : ASRCorruptor,
    gt_dict          : Dict[str, List[str]],
    targets          : List[float]      = None, 
    reps             : int              = 5,     # Number of times to repeat w different seeds
    dropout_schedule : Callable         = None,
    attempts         : int              = 10,
    seed             : int              = 0,
) -> Dict[str, Dict[str, List[str]]]:
    # Setup
    if targets          is None: targets          = [round(0.05 * i, 2) for i in range(1, 16)]
    if dropout_schedule is None: dropout_schedule = default_dropout_schedule

    synthetic_system_dicts = {}
    rng = random.Random(seed)

    # --------------------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------------------
    pbar_outer = tqdm(targets, desc="Generating transcripts", leave=True)
    for w in pbar_outer:
        drop_frac = dropout_schedule(w)

        pbar_inner = tqdm(range(reps), desc=f"WER={w:.2f}", leave=False)
        for r in pbar_inner:
            # Source IDs: "Synth_W{WER*100:02d}_r{rep_index}" (e.g. "Synth_W25_r0")
            src_name = f"Synth_W{int(w * 100):02d}_r{r}"
            synthetic_system_dicts[src_name] = {}

            for pid, ref_sents in gt_dict.items():
                # Join sentences if given as a list
                text_input = " ".join(ref_sents) if isinstance(ref_sents, list) else ref_sents

                # Generate synthetic output
                out = corruptor.generate(
                    text         = text_input,
                    target_wer   = w,
                    dropout_prob = drop_frac,
                    seed         = rng.randrange(1, 10**18),
                    attempts     = attempts,
                )

                # Wrap in list to match the standard src_sentence_list structure
                synthetic_system_dicts[src_name][pid] = [out["text"]]

        pbar_inner.close()
    pbar_outer.close()

    # Return: srcID -> {pID: [synthetic_transcript_str]}
    return synthetic_system_dicts

