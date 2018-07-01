import numpy as np
import sys
from utils.datareader import Datareader
from utils.definitions import *
from utils.evaluator import Evaluator
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.s_plus import tversky_similarity
import time
from utils.post_processing import *
from scipy import sparse
from utils.pre_processing import *
from utils.submitter import Submitter
from fbpca import pca
from fbpca import set_matrix_mult

datareader = Datareader(mode='offline', only_load=True)
evaluator = Evaluator(datareader)
urm = datareader.get_urm()
icm = datareader.get_icm(arid=True)

print('PCA...')
u, s, v = pca(icm, k=100)

print('Dot...')

icm_new = set_matrix_mult()
print(icm_new)
