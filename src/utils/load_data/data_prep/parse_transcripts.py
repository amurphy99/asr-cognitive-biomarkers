"""
Load transcript lists for feature generation.
--------------------------------------------------------------------------------
`src.utils.load_data.data_prep.parse_transcripts`

Sets up all of the data we need to generate all three biomarkers (grammar
features for Altered Grammar, and sentence lists for Perplexity Difference and
Pragmatic Impairment).

"""
import numpy  as np
import pandas as pd
import os

from tqdm.auto import tqdm

# From this project
from    .prepare_utterances           import prepare_df_speech
from    .text_preprocessing           import split_and_keep
from ....core.grammar.grammar_wrapper import extract_content_grammar_features


# ================================================================================
# Load transcripts in
# ================================================================================
# Get transcript lists for a directory (csv / parquet / xlsx)
def build_transcript_list(path, use_ids=None, exts=(".csv", ".parquet", ".xlsx")):
    files = [f for f in os.listdir(path) if any(f.endswith(ext) for ext in exts)]
    if use_ids is not None:
        use_ids = set(use_ids)
        files   = [f for f in files if f.split(".")[0] in use_ids]

    # (file, file_type, path)
    return [(f, f.split(".")[-1], path) for f in files]

# Read transcript based on file type
def import_transcript(file_type):
    if   file_type == "parquet": return pd.read_parquet
    elif file_type == "xlsx"   : return pd.read_excel
    elif file_type == "csv"    : return pd.read_csv
    else: raise ValueError(f"Bad file type: {file_type}")

# Merge meta DataFrame
def merge_with_meta(df: pd.DataFrame, meta_df: pd.DataFrame, *, verbose: int = 0) -> pd.DataFrame:
    # Working copies
    meta_df = meta_df.copy()
    feat_df = df.copy()

    # Original shapes
    shape_meta = meta_df.shape
    shape_feat = feat_df.shape

    # Merge into one DF
    common = set(meta_df["pID"]).intersection(set(feat_df["pID"]))
    base   = pd.DataFrame({"pID": sorted(common)})
    out    = pd.merge(feat_df, meta_df, on="pID", how="left")
  
    # Check it out
    if verbose >= 1:
        print(f"pIDs in common: {len(common)}")
        print(f"Original feats shape:   {shape_feat}, original meta shape: {shape_meta}")
        print(f"Output DataFrame shape: {out.shape } ")

    return out


# ================================================================================
# Get Pragmatic Impairment & data for Altered Grammar
# ================================================================================
def get_text_biomarker_data(
    transcriptions : list[tuple[str, str, str]],  # Ex. ("adrso001", "parquet", path)
    meta_df        : pd.DataFrame,                # Meta data about the pIDs (MMSE/MoCA scores, diagnosis. etc.)
    source         : str = "src",                 # Source dataset (e.g., "ADReSS-M" or "Delaware")
    *,
    keep_id         : str | None = None,   # Speaker ID to keep utterances for; if None, keeps all 
    min_words       : int        = 1,      # Minimum words to count a sentence
    concat_all_text : bool       = True,   # Combine all sentences/utterances for Altered Grammar features
    split_sentences : bool       = False,  # Try splitting sentences again on a set of end-of-sentence punctuation
    verbose         : int        = 0,      # How much information to print about processes
):
    """
    Extract all required features for the text-based biomarkers.

    NOTE: `concat_all_text` was set to True in the original configuration.

    NOTE: The current setup expects things like `min_words` to really be handled
    by each biomarker on their own further downstream. This just makes sure that
    there is SOME text at all to use.
    """
    # Book-keeping
    altered_grammar_data = []
    all_sentences        = {}
    first                = True

    # Loop through all files, one pID at a time
    pbar = tqdm(range(len(transcriptions)), desc=f"[{source}] Extracting Text Features", leave=True)
    for file, file_type, path in transcriptions:
        # --------------------------------------------------------------------------------
        # Prepare the transcription
        # --------------------------------------------------------------------------------
        pID = file.split(".")[0]
        if source == "IU": pID = "_".join(pID.split("_")[:3]).lower()
        pbar.set_postfix_str(f"pID: {pID}")

        # Import the transcript
        df = import_transcript(file_type)(f"{path}/{file}")

        # Get full text (sometimes the columns come with different names)
        df, utterances = prepare_df_speech(df, keep_id=keep_id, min_words=min_words)
   
        # --------------------------------------------------------------------------------
        # Get Altered Grammar features
        # --------------------------------------------------------------------------------
        # Extracts features from all text together OR extracts features from individual utterances

        # One row of features for all text in the transcript
        if concat_all_text: 
            gram_row = sentence_grammar_features(pID, utterances)
            if len(gram_row) > 1: altered_grammar_data.append(gram_row)
        
        # Extract multiple rows of features (one for each utterance/sentence)
        else:
            # Try to split the utterances again if specified
            if split_sentences:
                sentences = []
                for utterance in utterances:
                    if (len(utterance.strip()) < 2): continue  # (skip if too short anyways)
                    sentences.extend(split_and_keep(utterance))
                utterances = sentences
            
            # Keep the utterances as is
            else: sentences = [utterance for utterance in utterances if (len(utterance.strip()) > 1)]

            # Extract features
            gram_rows = [sentence_grammar_features(pID, [sentence]) for sentence in sentences]
            gram_rows = [row for row in gram_rows if (len(row) > 1)] 
            altered_grammar_data.extend(gram_rows)

        # --------------------------------------------------------------------------------
        # Book-keeping
        # --------------------------------------------------------------------------------
        all_sentences[pID] = utterances
        pbar.update(1)
        if first: first = False

        # Logging (if verbose & if the very last file)
        if (verbose >= 2) and (file == transcriptions[-1][0]):
            print(f"Debugging -- returning the last dataframe only. File: {file}")
            return df, utterances

    pbar.close()

    # Turn into DataFrame, merge with meta data, and return
    gram = pd.DataFrame(altered_grammar_data)
    gram = merge_with_meta(gram, meta_df, verbose=verbose)
    if verbose >= 1: print(f"Altered Grammar: {gram.shape}")

    return gram, all_sentences


# --------------------------------------------------------------------------------
# Altered Grammar features from sentence(s)
# --------------------------------------------------------------------------------
def sentence_grammar_features(pID: str, sentences: list[str]) -> dict:
    """
    If using a single sentence, wrap it inside a list before calling this.
    """
    # Get Altered Grammar features from the list of sentences
    feature_dict, feature_array = extract_content_grammar_features(sentences)

    # Combine the features with the pID to create a row of data
    row = {"pID": pID, **feature_dict}
    return row
