"""
Create N-gram language models to use for perplexity calculations.
--------------------------------------------------------------------------------
`src.core.perplexity.create_lm`

"""
import pandas as pd

# NLTK
from nltk      import ConditionalFreqDist
from nltk.util import ngrams as to_ngrams

# From this project
from ...utils.logging.logging import BOLD, UNBOLD
from ...utils.misc.setup_nltk import MIN_UTT_WORDS_DEFAULT, NGRAM_N_DEFAULT, STOP_MODE
from ...utils.load_data.data_prep.text_preprocessing import clean_utterances

# Perplexity utilities
from .token_formatting import process_text_POS, add_sentence_padding, flatten


# ================================================================================
# Create ngrams for each given participant ID
# ================================================================================
def ngrams_from_transcripts(
    transcriptions : dict[str, pd.DataFrame],     # Dictionary with a DataFrame of sentences for each pID
    *,
    keep_id        : str = "PAR",                 # Speaker column value to use
    min_utt_words  : int = MIN_UTT_WORDS_DEFAULT, # Required number of words to be a full "sentence" (w BOS/EOS tokens)
    ngram_n        : int = NGRAM_N_DEFAULT,       # Default N to use for creating the n-grams
    stop_mode      : str = STOP_MODE,             # Tokenization mode (e.g., POS conversion, keep stop words, etc.)
    verbose        : int = 1,                     # Logging detail level
):
    """
    Given a list of transcriptions and meta data, create ngrams for each given participant ID.

    transcriptions => Dictionary of {"pID": transcript_df} where transcript_df is a DataFrame
                      with rows for each utterance/turn, along with a speaker label, in the 
                      transcript.  
    
    ngrams_dict    => Dictionary of: {pID: [ (w1,w2,w3), (w2,w3,w4), ... ] }
    """
    # Track list of pIDs who had no suitable tokens to use
    missing_ids = []
    
    # --------------------------------------------------------------------------------
    # 1) Prepare the transcript texts for n-gram analysis
    # --------------------------------------------------------------------------------
    ngrams_dict = {}

    # Loop through the transcript data for each pID
    for pID, df in transcriptions.items():
        # Validate the sentence DataFrame formatting
        if ("pID" not in df.columns) or ("full_text" not in df.columns):
            raise ValueError(f"Sentences DataFrame for {BOLD}{pID}{UNBOLD} incorrectly formatted: missing 'pID' or 'full_text' columns.") 
        
        # Only use data for the specified speaker
        if keep_id is not None: df = df[df["pID"] == keep_id]

        # Normalize the full utterance text column values and convert them to a list
        utterances = df["full_text"].dropna().astype(str).values
        utterances = clean_utterances(utterances, MIN_UTT_WORDS=min_utt_words)
        if len(utterances) == 0: missing_ids.append(pID); continue
        
        # Convert utterances from strings to lists of words
        utterances = [utterance.split(" ") for utterance in utterances]

        # --------------------------------------------------------------------------------
        # 2) Apply parts-of-speech & stopwords (if specified) + BOS/EOS padding
        # --------------------------------------------------------------------------------
        utterances = [process_text_POS    (utterance, stop_mode=stop_mode) for utterance in utterances]
        utterances = [add_sentence_padding(utterance, ngram_n  =ngram_n  ) for utterance in utterances]
        
        # --------------------------------------------------------------------------------
        # 3) Convert utterances into n-grams
        # --------------------------------------------------------------------------------
        # TODO: Why convert to a list here?
        utt_grams = [list(to_ngrams(tokens, ngram_n)) for tokens in utterances]

        # Save the n-grams for this participant
        ngrams_dict[pID] = flatten(utt_grams)

    # --------------------------------------------------------------------------------
    # 4) Logging
    # --------------------------------------------------------------------------------
    if verbose == 1:
        counts = [len(utt_grams) for (_, utt_grams) in ngrams_dict.items()]
        avg_ng = 0 if (len(counts) == 0) else (sum(counts) / len(counts))
        nl = "\n" if missing_ids else ""
        print(
            f"N-gram dictionary successfully generated for {BOLD}N={ngram_n}{UNBOLD}. {BOLD}{sum(counts):,}{UNBOLD} total ngrams.\n"
            f"{BOLD}{len(ngrams_dict)}{UNBOLD} valid pIDs, {BOLD}{len(missing_ids)}{UNBOLD} pIDs with missing sentences.\n"
            f"    -> min ngrams: {BOLD}{min(counts):,}{UNBOLD}, max ngrams: {BOLD}{max(counts):,}{UNBOLD}, average ngrams: {BOLD}{avg_ng:,.2f}{UNBOLD}"
            f"{nl}{'pIDs with missing sentences: ' + str(missing_ids) if missing_ids else ''}"
        )

    # Return the new ngrams dictionary
    return ngrams_dict


# ================================================================================
# Create language models using NLTK's conditional frequency dists 
# ================================================================================
def create_language_model(ngrams_list: list[list[str]]) -> ConditionalFreqDist:
    # Tuple like: (context, next_token), where context is n-1 tokens long
    tupled = [(ngram[:-1], ngram[-1]) for ngram in ngrams_list]
    cfd = ConditionalFreqDist(tupled)
    return cfd

# Create a ConditionalFreqDist for one condition ('Control' or 'ProbableAD')
def lm_for_condition(
    ngrams_dict : dict[str, list[list[str]]], 
    *, 
    meta_df     : pd.DataFrame,
    condition   : str,
    exclude_pID : str | None = None,
):
    # If also excluding a certain participant ID from the language model
    df = meta_df.copy()
    if exclude_pID is not None: df = df[df["pID"] != exclude_pID]
    
    # Collect ngrams for all pIDs with the condition
    ngrams_list = []
    for pID in df[df["dx"] == condition]["pID"].values:
        ngrams_list += ngrams_dict.get(pID, [])

    # Create and return the language model
    cfd = create_language_model(ngrams_list)
    return cfd

