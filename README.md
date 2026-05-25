# asr-cognitive-biomarkers
Code for analyzing text-based cognitive biomarkers under automatic speech recognition error.


All final analysis is ran from two Jupyter notebooks (from the `notebooks/` directory). The `src/` tree below includes all supporting code.

---

## Architecture

```diff
 src/
 │
 ├── core/
 │   │
+│   ├── asr_corrupter/                     # Synthetic ASR error simulator
 │   │   ├── types.py                       # EditStats + ErrorProfile dataclasses
 │   │   ├── alignment.py                   # Levenshtein DP alignment + WER stats
 │   │   ├── text_utils.py                  # Normalize / tokenize helpers
 │   │   ├── corruptor.py                   # Learn error patterns and simulate them
 │   │   └── generate.py                    # Generation helpers across WER targets
 │   │
+│   ├── coherence/                         # Pragmatic Impairment biomarker
 │   │   ├── pragmatic_impairment.py        # Sliding-window cosine similarity
 │   │   └── utils/                         # Helpers for processing the input text
 │   │
+│   ├── grammar/                           # Altered Grammar biomarker
 │   │   ├── grammar_features.py            # Grammar-based feature extraction
 │   │   ├── grammar_wrapper.py             # High-level wrapper
 │   │   ├── feature_extraction/            # POS counts, lexical richness, syllables, etc.
 │   │   └── grammar_modeling/              # ML models, cross-validation, OOF prediction
 │   │
+│   └── perplexity/                        # Perplexity Difference biomarker
 │       ├── create_lm.py                   # Build / load n-gram language models
 │       ├── token_formatting.py            # POS / token formatting for LM input
 │       └── ngram_perplexity.py            # N-gram perplexity calculation
 │
 └── utils/
     │
     ├── config.py                          # Project-wide paths + constants
     │
     ├── load_data/                         # Data preprocessing
     │   ├── load_by_source.py              # Load in the ASR transcripts
     │   ├── label_speakers.py              # Levenshtein-based alignment for speaker labels
     │   └── data_prep/
     │       ├── text_preprocessing.py      # Normalize, tokenize, clean utterances
     │       └── parse_transcripts.py       # Parse raw transcript formats
     │
+    ├── analysis/                          # Quantitative analysis
     │   ├── word_error_rate.py             # Per-participant WER (via jiwer)
     │   ├── meta_analysis.py               # Biomarker stability across WER levels
     │   └── meta_correlation_table.py      # Biomarker correlation & meta-correlation
     │
     ├── figures/                           # Visualizations
     │   ├── paper/                         # Figures incorporated into the paper
     │   └── general/                       # Exploratory plots
     │
     ├── logging/
     │   └── logging.py                     # Console color + formatting helpers
     │
     └── misc/
         └── setup_nltk.py                  # NLTK initialization
```

