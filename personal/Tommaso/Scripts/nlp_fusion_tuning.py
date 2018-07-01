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
from recommenders.nlp import NLP
from utils.sparse import *

# INITIALIZATION
dr = Datareader(mode='offline', verbose=False, only_load=True)
ev = Evaluator(dr)
test_pids = dr.get_test_pids()
urm = dr.get_urm()
urm.data = np.ones(len(urm.data))

topk = 750

# nlp_strict = NLPStrict(dr)
# ucm_strict = nlp_strict.get_UCM().astype(np.float64)
# top_pop = dr.get_eurm_top_pop()
#
# # Do not train on challenge set
# ucm_strict_T = ucm_strict.copy()
# inplace_set_rows_zero(ucm_strict_T, test_pids)
# ucm_strict_T = ucm_strict_T.T
#
# sim = tversky_similarity(ucm_strict, ucm_strict_T, k=450, alpha=0.2, beta=0.5,
#                          shrink=150, target_items=test_pids)
#
# # Compute eurm
# eurm = dot_product(sim, urm, k=topk)
# eurm = eurm.tocsr()
# eurm = eurm[test_pids, :]

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

nlp = NLP(dr)

ucm = nlp.get_UCM(data1=data1).astype(np.float64)

# Do not train on challenge set
ucm_T = ucm.copy()
inplace_set_rows_zero(ucm_T, test_pids)
ucm_T = ucm_T.T

sim_lele = tversky_similarity(ucm, ucm_T, k=200, alpha=0.9, beta=1.0,
                              shrink=0, target_items=test_pids)

# Compute eurm
eurm_lele = dot_product(sim_lele, urm, k=topk)
eurm_lele = eurm_lele.tocsr()
eurm_lele = eurm_lele[test_pids, :]

# a = 0.2
# eurm_l1 = norm_l1_row(eurm)
# eurm_lele_l1 = norm_l1_row(eurm_lele)
# nlp_fusion = a * eurm_l1 + (1.0 - a) * eurm_lele_l1

#sparse.save_npz('nlp_fusion_tuned_online.npz', nlp_fusion)
# rec_list = eurm_to_recommendation_list(nlp_fusion, datareader=dr)
# ev.evaluate(rec_list, name='nlp_fusion_l1_a=' + str(a) + '_top_pop')
rec_list = eurm_to_recommendation_list(eurm_lele, datareader=dr)
ev.evaluate(rec_list, name='nlp_lele')

exit()



