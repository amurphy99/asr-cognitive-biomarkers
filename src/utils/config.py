"""
Define some configuration constants used throughout other files.
--------------------------------------------------------------------------------
`src.utils.config`

NOTE: The "Altered Grammar" biomarker usually has two results: one from a 
classifier model and one from a regressor model. The primary one from this paper
is the classifier, and the regressor is just kept as "gram_r".

"""
# Font to use for all text across a figure
FONT_FAMILY = "Times New Roman"

# --------------------------------------------------------------------------------
# Biomarker Name Conventions
# --------------------------------------------------------------------------------
BIOMARKER_COLS  = ("PragImp", "AltGram", "Perplex", "Gram_r") # "Gram_r"

# Rename base result column for each biomarker to slightly nicer versions
RENAME_MAP      = {"gc": "PragImp", "P_diff": "Perplex", "cls": "AltGram", "reg": "Gram_r"}

# Full biomarker names
BIOMARKER_NAMES = {
    "PragImp": "Pragmatic Impairment",
    "AltGram": "Altered Grammar",
    "Perplex": "Perplexity Difference",
    "Gram_r" : "AltGram Regressor",
}

# Some plots use just the short names + perplex=>perplexity
BIOMARKER_MAP_SHORT  = {
    "PragImp": "PragImp",
    "AltGram": "AltGram",
    "Perplex": "Perplexity",
}


# --------------------------------------------------------------------------------
# Transcript Source Names
# --------------------------------------------------------------------------------
SOURCE_NAMES = [
    "Hand", "Azure", "Google_v1",
    "WhisperLarge", "WhisperMedium", "WhisperSmall", "WhisperBase", "WhisperTiny",
]

# Map the transcript sources to shorter versions of their names
SRC_MAP_SHORT = {
    "Manually Annotated" : "Manual", "Hand": "Manual",
    "Azure"              : "Azure",  
    "Google_v1"          : "Google",
    "WhisperTiny"        : "W-Tiny", "Whisper-Tiny": "W-Tiny",
    "WhisperBase"        : "W-Base", 
    "WhisperSmall"       : "W-Small",
    "WhisperMedium"      : "W-Medium", 
    "WhisperLarge"       : "W-Large",
}

