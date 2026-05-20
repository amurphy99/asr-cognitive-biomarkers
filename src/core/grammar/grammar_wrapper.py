"""
Wrapper around the grammar features extraction,
--------------------------------------------------------------------------------
`src.core.grammar.feature_extraction.grammar_wrapper`

"""
from typing import Sequence, Union

# From this project
from ...utils.load_data.data_prep.text_preprocessing import preprocess
from .grammar_features import extract_altered_grammar_features

# Tokens that mark the end of a sentence (NLTK word_tokenize emits these standalone)
_SENTENCE_ENDERS = {".", "?", "!"}

# --------------------------------------------------------------------------------
# Cleaning / pre-processing
# --------------------------------------------------------------------------------
# Normalize whitespace inside one sentence string.
def _normalize_sentence_text(text: str) -> str:
    return " ".join(text.strip().split())

# Format sentence strings with ending punctuation if they don't have it already
def _check_sentence_boundary(text: str) -> str:
    # Guard for actual sentence content
    text = _normalize_sentence_text(text)
    if not text: return "" 

    # Return with the original punctuation or add a period if there wasn't any
    if text[-1] in _SENTENCE_ENDERS: return text
    return text + "."

# In the offline version we are given a list of sentences (sometimes), join them to match deployed.
def _sentences_to_cleaned(sentences: Union[str, Sequence[str]]) -> str:
    # If already given a string
    if isinstance(sentences, str): return _normalize_sentence_text(sentences)

    # Join sentences with a single whitespace character
    cleaned_sentences = [
        _check_sentence_boundary(sentence) for sentence in sentences if (sentence and sentence.strip())
    ]
    return " ".join(cleaned_sentences)


# ================================================================================
# Wrapper for the deployed signature (different inputs) matching offline version
# ================================================================================
def extract_content_grammar_features(sentences: list[str]) -> tuple[dict[str, float], list[float]]:
    # Pre-processing first (turns list of sentences into multiple)
    cleaned = _sentences_to_cleaned(sentences)
    if not cleaned: return {}, []

    # Tokenizes & gets parts-of-speech (POS) tags using NLTK
    cleaned, tokens, pos_tags = preprocess(cleaned)

    # Call the feature extraction function with the now matching inputs
    return extract_altered_grammar_features(cleaned=cleaned, tokens=tokens, pos_tags=pos_tags, words=None)
