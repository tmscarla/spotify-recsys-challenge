import numpy as np
from personal.Tommaso.NLP.GA_FeatureSelection import GA_FeatureSelection
from personal.Tommaso.NLP.NLP import NLP
from utils.datareader import Datareader
from utils.definitions import *
from utils.evaluator import Evaluator
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
import time
from recommenders.similarity.s_plus import dot_product, tversky_similarity, cosine_similarity
from utils.post_processing import *
from utils.pre_processing import *
from scipy import sparse
from recommenders.nlp import NLP
import gc


# INITIALIZATION
dr = Datareader(mode='offline', verbose=False, only_load=True)
ev = Evaluator(dr)
test_pids = dr.get_test_pids()
urm = dr.get_urm()
topk = 750

norm = True
work = True
split = True
skip_words = True
date = False
porter = False
porter2 = True
lanca = False
lanca2 = True
data1 = False

nlp = NLP(dr, stopwords=[], norm=norm, work=work, split=split, date=date, skip_words=skip_words,
                  porter=porter, porter2=porter2, lanca=lanca, lanca2=lanca2)

ucm = nlp.get_UCM(data1=data1)

# TVERSKY
for s in range(0, 200, 25):

    print('---------')
    print('TVERSKY | shrink =', s)

    sim = tversky_similarity(ucm, ucm.T, k=200, alpha=0.9, beta=1.0,
                             shrink=s, target_items=test_pids)

    # Compute eurm
    eurm = dot_product(sim, urm, k=topk)
    eurm = eurm.tocsr()
    eurm = eurm[test_pids, :]

    rec_list = eurm_to_recommendation_list(eurm, datareader=dr)

    ev.evaluate(rec_list, name='nlp_tversky_shrink=' + str(s))


