"""
Prepare tokens for perplexity (e.g., handle parts-of-speech, stopwords, etc.).
--------------------------------------------------------------------------------
`src.core.perplexity.token_formatting`

"""
from nltk import pos_tag

# From this project
from ...utils.misc.setup_nltk import TAGSET, STOP_MODE, STOP_SET, BOS_TOKEN, EOS_TOKEN

# --------------------------------------------------------------------------------
# Apply Parts-of-Speech (POS) & stopwords to given text
# --------------------------------------------------------------------------------
def process_text_POS(full_text: str | list[str], *, stop_mode : str = STOP_MODE) -> str | list[str]:
    """
    Replace each token with a part-of-speech tag, optionally keeping stopwords as tokens.
    """
    return_words = not (type(full_text) == str)
    if not return_words: words = full_text.split(" ")
    else:                words = full_text

    # Get POS tags
    tagged = pos_tag(words, tagset=TAGSET)

    # Keep stopwords if set to    
    if stop_mode == "keep":  words = [(word if (word in STOP_SET) else tag) for word, tag in tagged]
    else:                    words = [tag for _, tag in tagged]

    # Return in sentence (str) or list-of-words (List[str]) format
    if return_words: return words
    else:            return " ".join(words)

# --------------------------------------------------------------------------------
# Add Beginning-of-Sentence (BOS) and End-of-Sentence (EOS) padding tokens
# --------------------------------------------------------------------------------
def add_sentence_padding(full_text: str | list[str], *, ngram_n: int) -> str | list[str]:
    """
    Add <START>/<STOP> padding for n-grams.
    """
    return_words = not (type(full_text) == str)

    # Full text in string form
    if not return_words: 
        bos_pad = (BOS_TOKEN + " ") * (ngram_n - 1)
        eos_pad = (" " + EOS_TOKEN) * (ngram_n - 1)
        
    # Full text in list of words/tokens form
    else:
        bos_pad = ([BOS_TOKEN] * (ngram_n - 1))
        eos_pad = ([EOS_TOKEN] * (ngram_n - 1))

    return bos_pad + full_text + eos_pad

# --------------------------------------------------------------------------------
# Custom list flattening helper
# --------------------------------------------------------------------------------
def flatten(list_of_lists):
    """
    Flatten a list of lists once.
    """
    return [token for sub in list_of_lists for token in sub]
