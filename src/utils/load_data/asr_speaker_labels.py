"""
Create new versions of the ASR transcripts with reference-based speaker labels.
--------------------------------------------------------------------------------
`src.utils.load_data.asr_speaker_labels`

Using the hand-annotated transcripts as a reference, align each of the ASR
transcripts with the reference data and apply the speaker labels from the
reference. Allows us to exclude non-participant speech.

"""
import os
import pandas as pd
 
from tqdm.auto import tqdm

# From this project
from .load_by_source              import SourceConfig
from .label_speakers              import label_asr_speakers
from .data_prep.parse_transcripts import import_transcript


# ================================================================================
# Standardizes column names across transcript sources
# ================================================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical column names after normalization:
      speaker_id, word, s (start time), e (end time), uttID
    """
    column_maps = {
        "speaker_id": ["pID", "Speaker", "speaker", "speaker_id", "speaker_ID"],
        "word"      : ["word", "text", "content", "result"],
        "s"         : ["s", "s_ts", "start", "start_ts", "start_time", "t0"],
        "e"         : ["e", "e_ts", "end",   "end_ts",   "end_time",   "t1"],
        "uttID"     : ["uttID", "utteranceID", "uID"],
    }
    for canonical, variations in column_maps.items():
        for variation in variations:
            if variation in df.columns:
                df = df.rename(columns={variation: canonical})
                break
 
    return df
 
 
# ================================================================================
# Add a speaker column to & resave transcripts from each source
# ================================================================================
def add_speaker_labels_to_asr(
    data_sources: list[SourceConfig],  # List of SourceConfigs; data_sources[0] must be the "Hand" source
    skip_list   : list[str],           # List of pIDs to exclude entirely
    save_to     : str,                 # Root output directory; one sub-folder is created per source
    save        : bool = True,         # Whether or not to actually save
) -> None:
    """
    Aligns ASR transcripts with their hand-annotated reference to add speaker
    labels, then filters to participant-only speech (PAR) and saves each file
    as a .parquet.
 
    Expects data_sources[0] to be the hand-annotated (reference) source.
    All subsequent sources in the list are treated as ASR systems to process.
    Files that were already processed (exist in the expected directory) are
    skipped automatically.
    """
    # --------------------------------------------------------------------------------
    # Build reference transcript index from the hand-annotated source
    # --------------------------------------------------------------------------------
    # Load in all hand transcriptions
    hand_config = data_sources[0]
 
    # Get the list of files & turn that into a dictionary
    file_list = [x for x in os.listdir(hand_config.path) if x.endswith(".parquet") or x.endswith(".csv")]
    ref_transcripts = {
        file.split(".")[0]: (file, hand_config.ext, hand_config.path)
        for file in file_list if (file.split(".")[0] not in skip_list)
    }

    ref_keys = list(ref_transcripts.keys())
 
    # --------------------------------------------------------------------------------
    # Loop through each ASR source (skip the Hand source at index 0)
    # --------------------------------------------------------------------------------
    for config in data_sources[1:]:
 
        # Get files present in this source that also exist in the hand reference
        file_list   = [x for x in os.listdir(config.path) if x.endswith(".parquet") or x.endswith(".csv")]
        transcripts = [
            (file, config.ext, config.path)
            for file in file_list if (file.split(".")[0] in ref_keys)
        ]
 
        # Create output directory
        save_dir = f"{save_to}/{config.name}"
        os.makedirs(save_dir, exist_ok=True)
 
        # ============================================================================
        # Loop through all files, one pID at a time
        # ============================================================================
        pbar = tqdm(transcripts, desc=f"[{config.name}] Adding speaker labels", leave=True)
        for file, file_type, path in pbar:
            pID = file.split(".")[0]
            pbar.set_postfix_str(f"pID: {pID}")
 
            # Skip if already processed
            save_file = f"{save_dir}/{pID}.parquet"
            if os.path.exists(save_file): continue #pbar.update(1); continue
 
            # Skip if no reference transcript
            if pID not in ref_keys: continue
            ref_file, ref_type, ref_path = ref_transcripts[pID]
 
            # ------------------------------------------------------------------------
            # Load and normalize both transcripts
            # ------------------------------------------------------------------------
            # Import and prepare ASR transcript
            asr_df = import_transcript(file_type)(f"{path}/{file}")
            asr_df = _normalize_columns(asr_df)
 
            # Import and prepare the reference transcript
            ref_df = import_transcript(ref_type)(f"{ref_path}/{ref_file}")
            ref_df = _normalize_columns(ref_df)
 
            # ------------------------------------------------------------------------
            # Assign speaker labels and keep only participant speech
            # ------------------------------------------------------------------------
            labeled = label_asr_speakers(ref_df=ref_df, asr_df=asr_df, ref_speaker_col="speaker_id")
 
            # Change the label to the new one we have
            asr_df["speaker_id"] = labeled["speaker_true"].values
            asr_df = asr_df[asr_df["speaker_id"] == "PAR"]
 
            # Save the file
            if save: asr_df.to_parquet(save_file)
 
        pbar.close()

