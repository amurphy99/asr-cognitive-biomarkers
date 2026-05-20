"""
Load data for biomarker generation for all given transcript sources. 
--------------------------------------------------------------------------------
`src.utils.load_data.load_by_source`

NOTE: Some of the text normalization for sentences happens here as well.

"""
import pandas as pd

import os, re

from dataclasses import dataclass
from tqdm.auto   import tqdm

# From this project
from .data_prep.parse_transcripts import get_text_biomarker_data


# --------------------------------------------------------------------------------
# Configuration dataclass to organize the different info for each transcript set
# --------------------------------------------------------------------------------
@dataclass
class SourceConfig:
    name        : str           # label, goes into srcID
    path        : str           # directory with transcripts
    ext         : str           # ".csv" / ".xlsx" / ".parquet"
    meta        : pd.DataFrame  # which meta DF to use (adr, d0, pt_merge, etc.)
    meta_id_col : str           # which meta col matches file base name
    use_ids     : list          # list of IDs to keep (e.g., use_transcriptions or use_trans_pitt)


# ================================================================================
# Load in the transcripts from each provided ASR source
# ================================================================================
def load_data_by_source(data_sources: list[SourceConfig]) -> dict:
    """
    Loads transcripts for each source in data_sources, extracts text biomarker
    features, and cleans the sentence-level text.
    """
    # Store resulting features for each transcript source
    data_by_source = {}
 
    # Loop through the configs for each data source 
    pbar = tqdm(data_sources, desc="Preparing data", leave=True)
    for config in pbar:
        pbar.set_description(f"Preparing data from source: {config.name}")
 
        # --------------------------------------------------------------------------------
        # Gather transcript file list for this source
        # --------------------------------------------------------------------------------
        # Proper file extension
        file_list = [x for x in os.listdir(config.path) if x.endswith(".parquet") or x.endswith(".csv")]

        # Not in the list of pIDs we want to exclude
        transcripts = [(file, config.ext, config.path) for file in file_list if (file.split(".")[0] in config.use_ids)]
   
        # --------------------------------------------------------------------------------
        # Extract biomarker features and raw sentence lists
        # --------------------------------------------------------------------------------
        alt_gram_df, all_sentences = get_text_biomarker_data(
            transcripts,
            meta_df   = config.meta,
            source    = config.name,
            keep_id   = "PAR",  # "PAR" | None | Speaker ID to keep utterances for; if None, keeps all 
            min_words = 0,      # Minimum words to count a sentence
            verbose   = 0,
        )

        # --------------------------------------------------------------------------------
        # Loop through every pID and sentence list in the dictionary
        # --------------------------------------------------------------------------------
        updated_sentences = {}
        for pID, sentence_list in all_sentences.items():

            # Clean the sentences
            new_sentences = [re.sub(r"[^\w\s]", "", sentence.lower()) for sentence in sentence_list]
    
            # Save the updated list back under the same key
            updated_sentences[pID] = new_sentences
 
        # --------------------------------------------------------------------------------
        # Store results
        # --------------------------------------------------------------------------------
        data_by_source[config.name] = {
            "alt_gram" : alt_gram_df,        # altered grammar features
            "sentences": updated_sentences,  # cleaned sentence lists per participant
        }
 
    pbar.close()
    return data_by_source
 