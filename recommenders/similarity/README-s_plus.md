"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

GUIDE FOR THE S_PLUS

IMPORTANT NOTES: 
    - each similarity keep top k elements per row.
    - all the matrixes returned are in coo format with already the zero elements removed.

The s_plus package contains the following functions:

1. SIMILARITY BASE
- dot_product_similarity()

2. SIMILARITIES WITH NORMALIZATION
- cosine_similarity()
- tversky_similarity()
- jaccard_similarity()
- dice_similarity()

3. SIMILARITIES WITH WEIGHTS
- feature_weight_similarity()
- popularity_weight_similarity()
- popularity_feature_weight_similarity()

4. STOCHASTIC SIMILARITIES AND RATING
- p3alpha_similarity()
- rp3beta_eurm()

5. SIMILARITIES WITH WEIGHTS AND NORMALIZATION
- s_plus_similarity()

6. OTHER FUNCTIONS
- dot_product()
- s_plus()


COMPLETE FUNCTIONS AND PARAMETERS LIST

1. SIMILARITY BASE

def dot_product_similarity(m1, m2, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0)
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

2. SIMILARITIES WITH NORMALIZATION

def cosine_similarity(m1, m2, alpha=0.5, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    alpha: weight normalization terms of m1 with alpha and weight normalization terms of m2 with 1-alpha
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute
    --- NOTE: alpha=0.5 it's the symmetric cosine similarity, chage this value to get asymmetric cosine ---

def tversky_similarity(m1, m2, alpha=1, beta=1, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    alpha: change the weight of the norm terms of m1
    beta: change the weigh of the norm terms of m2
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute
    --- NOTE: jaccard similarity is tversky similarity with alpha=1 and beta=1 ---
    --- NOTE: dice similarity is tversky similarity with alpha=0.5 and beta=0.5 ---
    --- NOTE: tanimoto similarity is tversky similarity with alpha=1, beta=1 and binary=False ---

def jaccard_similarity(m1, m2, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

def dice_similarity(m1, m2, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

3. SIMILARITIES WITH WEIGHTS

NOTE:
    - feature weights -> high value INCREASE similarities score
    - popularity weights -> high value DECREASE similarities score
    - possible values for weights array:
        - array: array of weights values
        - 'none': don't use weights (weights equal to one)
        - 'sum': use the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)
        - 'log': use the log (base 10) of the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)
        - 'ln': use the ln (base e) of the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)


def feature_weight_similarity(m1, m2, weight_feature_m1='sum', weight_feature_m2='sum', w1=1, w2=1, k=100, shrink=0, threshold=0,binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    weight_feature_m1: feature weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_feature_m1: feature weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    w1: coefficient that power the feature weights array of m1
    w2: coefficient that power the feature weights array of m2
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

def popularity_weight_similarity(m1, m2, weight_pop_m1='sum', weight_pop_m2='sum', p1=1, p2=1, k=100, shrink=0, threshold=0,binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    weight_pop_m1: popularity weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_pop_m2: popularity weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    p1: coefficient that power the popularity weights array of m1
    p2: coefficient that power the popularity weights array of m2
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

def popularity_feature_weight_similarity(m1, m2, weight_pop_m1='sum', weight_pop_m2='sum', weight_feature_m1='sum' , weight_feature_m2='sum', p1=1, p2=1, w1=1, w2=1, k=100, shrink=0, threshold=0,binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    weight_pop_m1: popularity weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_pop_m2: popularity weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_feature_m1: feature weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_feature_m1: feature weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    p1: coefficient that power the popularity weights array of m1
    p2: coefficient that power the popularity weights array of m2
    w1: coefficient that power the feature weights array of m1
    w2: coefficient that power the feature weights array of m2
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute


4. STOCHASTIC SIMILARITIES AND RATING

def p3alpha_similarity(m1, m2, weight_pop_m1='sum' , weight_pop_m2='sum', alpha = 1, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    weight_pop_m1: let 'sum' automatically build the probabilities for the matrix m1, 'none' if you already have the probability matrix
    weight_pop_m2: let 'sum' automatically build the probabilities for the matrix m2, 'none' if you already have the probability matrix
    alpha: coefficient that element wise power the similarity matrix
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

def rp3beta_eurm(urm, p3alpha_similarity, k=100, shrink=0, threshold=0, weight_pop=None, beta=1, binary=False, target_items=None, verbose=0):
    urm: urm sparse matrix (NxM)
    p3alpha_similarity: p3alpha similarity sparse matrix (MxM)
    weight_pop: array (1xM) with popularities of each element
    beta: coefficient that power the popularities array
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute

5. SIMILARITIES WITH WEIGHTS AND NORMALIZATION

NOTE:
    - feature weights -> high value INCREASE similarities score
    - popularity weights -> high value DECREASE similarities score
    - possible values for weights array:
        - array: array of weights values
        - 'none': don't use weights (weights equal to one)
        - 'sum': use the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)
        - 'log': use the log (base 10) of the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)
        - 'ln': use the ln (base e) of the sum of the element for each row (popularity) or for each column (feature) of the matrix m1 (since m2 is transpose, it use columns for popularities and rows for features)

def s_plus_similarity(m1, m2, weight_pop_m1='sum' , weight_pop_m2='sum', weight_feature_m1='sum', weight_feature_m2='sum', p1=1, p2=1, w1=1, w2=1, normalization=True, l=0.5, c=0.5, t1=1, t2=1, k=100, shrink=0, threshold=0, binary=False, target_items=None, verbose=0):
    m1: first sparse matrix (NxM)
    m2: second sparse matrix (MxL)
    weight_pop_m1: popularity weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_pop_m2: popularity weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_feature_m1: feature weights array (1xN) for m1 (possible values: array, 'none', 'sum', 'log', 'ln')
    weight_feature_m1: feature weights array (1xL) for m2 (possible values: array, 'none', 'sum', 'log', 'ln')
    p1: coefficient that power the popularity weights array of m1
    p2: coefficient that power the popularity weights array of m2
    w1: coefficient that power the feature weights array of m1
    w2: coefficient that power the feature weights array of m2
    normalization: if True use the normalization terms to get value between 0 and 1 using mix of tversky and cosine
    l: weight term between [0,1], near 1 more weight to tversky term (l), near 0 more weight to cosine term (1-l)
    c: weight cosine normalization terms of m1 with alpha and weight cosine normalization terms of m2 with 1-c
    t1: change the weight of the tversky norm terms of m1
    t2: change the weigh of the tversky norm terms of m2
    k: top k items per row
    shrink: shrink term
    threshold: all the values under this value are cutted
    binary: True use the real values in the input matrix, False use binary values (0 or 1)
    target_items: compute only the rows that you need (if setted to None it compute the whole matrix)
    verbose: 1 show progress, 0 mute


6. OTHER FUNCTIONS
def dot_product():
def s_plus():
    this function include all the function above, you can set each coefficient (so sometimes is not more a similarity)