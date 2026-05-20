"""
Functions for calculating perplexity from given ngrams.
--------------------------------------------------------------------------------
`src.core.perplexity.ngram_perplexity`

"""
import math

# From this project
from ...utils.misc.setup_nltk import V_POS_STOP

# Tracking how many ngrams were "unseen" / use the default smoothing probability
not_seen, yes_seen = 0, 0

# ================================================================================
# Calculate perplexity for a given list of ngrams
# ================================================================================
def calculate_perplexity(test_ngrams, cfd, ngram_n, smoothing=True, V=V_POS_STOP, alpha=1, return_ln=False):
    """
    Calculate perplexity for a given list of ngrams.
    ---
    Params:
     test_ngrams => List of ngrams to calculate perplexity for
     cfd         => ConditionalFreqDist
     ngram_n     => N of the given ngrams (1=unigrams, 2=bigrams, 3=trigrams, etc.)
     smoothing   => If Laplace smoothing should be used
     V           => Vocabulary size (need this for smoothing)
     alpha       => Laplace smoothing constant
     return_ln   => Return the average ln_probability instead of the perplexity
    """
    # These are for the individual ngram calculations
    args = dict(cfd=cfd, ngram_n=ngram_n, smoothing=smoothing, V=V, alpha=alpha)
    
    # Calculate ln probabilities for each individual ngram in the given test set
    sum_of_ln_probabilities = 0
    for test_ngram in test_ngrams:   
        ln_probability = calc_ln_probability(test_ngram, **args)
        sum_of_ln_probabilities += ln_probability
            
    # Get the average ln probability
    average = sum_of_ln_probabilities / (len(test_ngrams))
    
    # Return either the average ln probability itself, or finish the perplexity calculation and return that
    if return_ln: return           average
    else:         return math.exp(-average) 


# ================================================================================
# Calculate the ln_probability for an individual ngram
# ================================================================================
def calc_ln_probability(ngram, cfd, ngram_n, smoothing=False, V=None, alpha=1):
    global not_seen
    global yes_seen
    
    # --------------------------------------------------------------------------------
    # Get counts and total possible outcomes (before smoothing)
    # --------------------------------------------------------------------------------
    # Unigrams
    if ngram_n == 1: 
        count = cfd[ngram]
        total_possible_outcomes = sum(cfd.values())
    
    # Bigrams
    elif ngram_n == 2: 
        count = cfd[ngram[0]][ngram[1]]
        total_possible_outcomes = sum(cfd[ngram[0]].values())
    
    # All ngrams where n > 2
    else: 
        ngram_condition         = tuple(ngram[:-1]) # tuple([word for word in ngram[:-1]])
        count                   =     cfd[ngram_condition][ngram[-1]]
        total_possible_outcomes = sum(cfd[ngram_condition].values())
        
    # --------------------------------------------------------------------------------
    # Calculations (smoothing/guarding)
    # --------------------------------------------------------------------------------
    # Tracking how many ngrams were "unseen"
    if total_possible_outcomes == 0: not_seen += 1
    else:                            yes_seen += 1
    
    # Calculate the probability (adjust values if smoothing is enabled)
    if smoothing:
        count                   = count                   +      alpha
        total_possible_outcomes = total_possible_outcomes + (V * alpha)
    
    # Without smoothing we have to guard for an unknown context
    elif (count == 0) or (total_possible_outcomes == 0): return 0.0
    
    # Calculate final probability and ln_probability
    probability    = count / total_possible_outcomes
    ln_probability = math.log(probability)
    
    return ln_probability
