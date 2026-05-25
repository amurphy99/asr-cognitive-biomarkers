"""
Text normalization, tokenization, and segmentation helpers for error simulation.
--------------------------------------------------------------------------------
`src.core.error_synthesis.text_utils`

"""
import re
from typing import List

# From this project
from .types import Sentence


# Keep apostrophes
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


# --------------------------------------------------------------------------------
# Normalization / tokenization
# --------------------------------------------------------------------------------
def normalize(text: str) -> str:
    """
    Lowercase and strip non-alphanumeric characters.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())

def tokenize(text: str) -> Sentence:
    """
    Split normalized text into a list of word tokens.
    """
    return _WORD_RE.findall(text)

# --------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------
def segment(text: str, mode: str = "newline") -> List[str]:
    """
    Split a long text block into utterance-level segments.
    """
    text = text.strip()
    if not text: return []

    if   mode == "newline"  : return [line.strip() for line in text.splitlines()          if line.strip()]
    elif mode == "paragraph": return [   p.strip() for p    in re.split(r"\n\s*\n", text) if    p.strip()]
    elif mode == "none"     : return [text]
    else                    : raise ValueError(f"Unknown segment mode: {mode}")

