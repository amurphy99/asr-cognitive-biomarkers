"""
Prepare input DataFrame for grammar feature extraction.
--------------------------------------------------------------------------------
`src.utils.load_data.data_prep.prepare_utterances`

"""
import numpy  as np
import pandas as pd

# From this project
from .text_preprocessing import clean_sentences


# ================================================================================
# Fully prepare input DataFrame 
# ================================================================================
def prepare_df_speech(
    df: pd.DataFrame,               # Speech/transcription DataFrame
    *, 
    keep_id   : str | None = None,  # ID of the speaker (keep all utterances/sentences if None)
    min_words : int        = 0,     # Minimum words to count as a full sentence
) -> tuple[pd.DataFrame, list[str]]:
    # Add any needed columns for full utterances and speaker IDs
    # (final cols needed are: "pID" & "full_text")
    df = _add_full_utterance_columns(df)
    df = _normalize_speaker_column  (df)

    # Only use data for the specified speaker
    if (keep_id is not None): df = df[df["pID"] == keep_id]

    # DataFrame still in one-row-per-word format, reduce to one per utterance
    df  = df.drop_duplicates(subset="uttID", keep="last")

    # --------------------------------------------------------------------------------
    # Turn DataFrame into list of sentences
    # --------------------------------------------------------------------------------
    # Convert to a regular list before cleaning up known inconsistencies
    sentences = df["full_text"].to_list()

    # Clean up various artifacts present in the given list of sentences
    sentences = clean_sentences(sentences)

    # Filter for a minimum word count
    sentences = [x for x in sentences if ((type(x) == str) and (len(x.split(" ")) > min_words))]
    
    # Return both the DataFrame and the cleaned sentences list
    return df, sentences


# --------------------------------------------------------------------------------
# Helpers for files with no "uttID" but have "full_text"
# --------------------------------------------------------------------------------
# Make sure there are "full_text" & "uttID" columns to use
# TODO: Need to use the column normalization function I defined in the notebook
def _add_full_utterance_columns(df: pd.DataFrame):
    # If "uttID" column exists, make full utterance from that
    if "uttID" in df.columns:
        use_col = "word" if "word" in df.columns else "speech"
        df["full_text"] = df.groupby("uttID")[use_col].transform(lambda x: " ".join(x))
        return df
    
    # Check for different column names
    cols = df.columns
    if   "Speech"        in cols: df = df.rename(columns={"Speech": "full_text"})
    elif "speech"        in cols: df = df.rename(columns={"speech": "full_text"})
    elif "full_text" not in cols: df["full_text"] = df.groupby("uttID")["word"].transform(lambda x: " ".join(x))

    # Add an "uttID" column if not there
    if "uttID" not in cols: df = _add_uttID_df(df)
    return df

# Helper for files with no "uttID" but have "full_text"
def _add_uttID_df(df: pd.DataFrame) -> pd.DataFrame:
    new_text      = (df["full_text" ] != df["full_text" ].shift(1))
    new_speaker   = (df["speaker_id"] != df["speaker_id"].shift(1))
    new_utterance = (new_text | new_speaker).fillna(True)
    
    df["uttID"] = new_utterance.astype(int).cumsum()
    return df

# --------------------------------------------------------------------------------
# Change speaker column name to "pID"
# --------------------------------------------------------------------------------
def _normalize_speaker_column(df: pd.DataFrame) -> pd.DataFrame:
    variations = ["pID", "Speaker", "speaker", "speaker_id", "speaker_ID"]
    for variation in variations:
        if variation in df.columns: df = df.rename(columns={variation: "pID"})
    return df

