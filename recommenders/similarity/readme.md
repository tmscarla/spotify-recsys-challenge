"""
    author: Simone Boglio
    mail: simone.boglio@mail.polimi.it
"""

REQUIREMENTS:
    gcc, cython

INSTALLATION:
    to compile all the cyton file run this command in this folder:
    python compileCython.py build_ext --inplace

NOTE: 
    if you get the following warning don't worry, it's ok 
    --> cc1plus: warning: command line option '-Wstrict-prototypes' is valid for C/ObjC but not for C++

    if you have problem in compiling with spacename try to rename the __init__.py file with another name, next try to compile again, at the end you must rename the __init__.py file or the imports could fail.

    all the time the package is updated you must run again the compile script since you must regenerate the new code
    
    
    
    
SIMILARITY PACKAGE

!!!ALL THE FUNCTIONS RETURN A SPARSE MATRIX IN COO FORMAT!!!

dot_product:
    def dot_product(a, b, k=100)
        a: first sparse matrix (NxM)
        b: second sparse matrix (MxR)
        k: knn per row
    def dot_product_similarity(items, k=100)
        items: sparse matrix (NxM)
        k: knn per row

cosine:
    def cosine_similarity(items, alpha=0.5, k=100, shrink=0, threshold=0, binary=True)
        items: sparse matrix
        alpha: weight norm terms of the rows with alpha and weight norm terms of the cols with 1-alpha
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        binary: True use the real values in the input matrix, False use binary values (0 or 1)
        --- NOTE: alpha=0.5 it's the symmetric cosine similarity, chage this value to get asymmetric cosine
        
jaccard:
    def jaccard_similarity(items, k=100, shrink=0, threshold=0, binary=False)
        items: sparse matrix
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        binary: True use the real values in the input matrix, False use binary values (0 or 1)
        
    def tanimoto_similarity(items, k=100, shrink=0, threshold=0)
        items: sparse matrix
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        --- NOTE: tanimoto is jaccard binary ---
        
dice:
    def dice_similarity(items, k=100, shrink=0, threshold=0, binary=False)
        items: sparse matrix
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        binary: True use the real values in the input matrix, False use binary values (0 or 1)
        
tversky:
    def tversky_similarity(items, alpha=1, beta=1, k=100, shrink=0, threshold=0, binary=False)
        items: sparse matrix
        alpha: change the weight of the norm terms of the rows
        beta: change the weigh of the norm terms of the cols
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        binary: True use the real values in the input matrix, False use binary values (0 or 1)
        --- NOTE: jaccard similarity is tversky similarity with alpha=1 and beta=1 ---
        --- NOTE: dice similarity is tversky similarity with alpha=0.5 and beta=0.5 ---
        --- NOTE: tanimoto similarity is tversky similarity with alpha=1, beta=1 and binary values ---

p3alpha_rp3beta:
    def p3alpha_rp3beta_similarity(items, users, popularities, alpha=1, beta=1, k=100, shrink=0, threshold=0)
        items: sparse matrix (NxM)
        users: sparse matrix (MxN)
        popularities: array with popularities of the items (N)
        alpha: change weight of the numerator before apply the norm terms
        beta: change the weight of the norm terms of the cols
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted

    def rp3beta_similarity(items, users, popularities, beta=1, k=100, shrink=0, threshold=0)
        items: sparse matrix (NxM)
        users: sparse matrix (MxN)
        popularities: array with popularities of the items (N)
        beta: change the weigh of the norm terms of the cols
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        --- NOTE: rp3beta is rp3alpha_rp3beta with alpha=1 ---

    def p3alpha_similarity(items, users, alpha=1, k=100, shrink=0, threshold=0)
        items: sparse matrix (NxM)
        users: sparse matrix (MxN)
        alpha: change weight of the similarity value
        k: knn per row
        shrink: shrink term
        threshold: all the similarity values under this value are cutted
        --- NOTE: p3alpha is rp3alpha_rp3beta with beta=0 and popularities all equal to 1 ---
        
utility:
    def estimate_memory_matrix_product(matrix, k)
        matrix: sparse matrix
        k: knn per row 
        -> estimate the size in ram necessary to compute the matrix product with k value per row
        
    def max_k_matrix_product(matrix)
        matrix: sparse matrix
        -> return the max value of k that you could ask for a matrix prodcut and also the estimated size in ram with the max k
        
        
        
        
        
        
        
        