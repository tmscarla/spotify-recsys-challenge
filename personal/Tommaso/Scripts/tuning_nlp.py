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
from recommenders.nlp_strict import NLPStrict


# INITIALIZATION
dr = Datareader(mode='offline', verbose=False, only_load=True)
ev = Evaluator(dr)
test_pids = dr.get_test_pids()
urm = dr.get_urm()
topk = 750

nlp_strict = NLPStrict(dr)
ucm_strict = nlp_strict.get_UCM()

# TVERSKY
for a in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.7, 2.0]:

    print('---------')
    print('TVERSKY | power =', a)

    sim = tversky_similarity(ucm_strict, ucm_strict.T, k=450, alpha=0.2, beta=0.5,
                             shrink=150, target_items=test_pids)

    sim.data = np.power(sim.data, a)

    # Compute eurm
    eurm = dot_product(sim, urm, k=topk)
    eurm = eurm.tocsr()
    eurm = eurm[test_pids, :]

    rec_list = eurm_to_recommendation_list(eurm, datareader=dr)

    ev.evaluate(rec_list, name='nlp_strict_tversky_power=' + str(a))

