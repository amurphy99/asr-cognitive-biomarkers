"""
Setup for NLTK tokenizers/other stuff.
--------------------------------------------------------------------------------
`src.utils.misc.setup_nltk`

"""
import nltk
from nltk          import word_tokenize, pos_tag
from nltk.util     import ngrams
from nltk.corpus   import stopwords, cmudict
from nltk.tokenize import SyllableTokenizer


# ================================================================================
# Make sure the right stuff is installed
# ================================================================================
# Make sure the resources exist (works on most NLTK versions)
def _check_nltk(universal: bool = False) -> None:
    """
    One-time NLTK setup. Safe to call multiple times.
    Ensures:
      - punkt tokenizer
      - stopwords
      - POS tagger (averaged_perceptron_tagger[_eng])
      - optional universal_tagset
    """
    print("Checking for nltk installation...")
    needed = {
        "tokenizers/punkt"     : "punkt",
        "tokenizers/punkt_tab" : "punkt_tab",
        "corpora/stopwords"    : "stopwords",
        "corpora/cmudict"      : "cmudict",
    }

    # Core resources
    for probe, pkg in needed.items():
        try:                nltk.data.find(probe);          print(f"{probe} found")
        except LookupError: nltk.download(pkg, quiet=True); print(f"{probe} downloaded from {pkg}")

    # POS tagger (name varies across versions)
    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{res}")
            print(f"taggers/{res} found")
            break
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                print(f"taggers/{res} downloaded")
                break
            except Exception: pass

    # Optional universal tagset
    if universal:
        try: nltk.data.find("taggers/universal_tagset")
        except LookupError:
            try: nltk.download("universal_tagset", quiet=True)
            except Exception: pass

# Call once at import
_check_nltk()

# ================================================================================
# Perplexity Configuration 
# ================================================================================
NGRAM_N_DEFAULT       = 3
MIN_UTT_WORDS_DEFAULT = 3

TAGSET    = None                            # None => Penn Treebank
STOP_MODE = "keep"                          # "keep" => keep stopword instead of POS tag
STOP_SET  = set(stopwords.words("english"))

BOS_TOKEN = "<START>"
EOS_TOKEN = "<STOP>"

LEN_UNIQUE_POS       = 36            # NLTK PTB POS tags
LEN_UNIQUE_STOPWORDS = len(STOP_SET)
V_POS_STOP           = LEN_UNIQUE_POS + LEN_UNIQUE_STOPWORDS  # default vocab size

