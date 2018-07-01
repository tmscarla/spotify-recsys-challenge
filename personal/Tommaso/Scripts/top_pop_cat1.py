import sys
from scipy import sparse
import numpy as np
import utils.pre_processing as pre
from utils.definitions import *
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import *
from utils.post_processing import *

dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)
urm = dr.get_urm(binary=True)
urm_csc = urm.tocsc(copy=True)

sim_nlp = sparse.load_npz(ROOT_DIR + '/data/sim_nlp_lele.npz')

for k in [1, 2, 3, 4, 5]:
    eurm_top = dr.get_eurm_top_pop_filter_cat_1(sim_nlp, k, topk=500)
    eurm_top = norm_l1_row(eurm_top)

    eurm_nlp = sparse.load_npz(ROOT_DIR + '/data/nlp_fusion_tuned_offline.npz')
    eurm_nlp = norm_l1_row(eurm_nlp)

    for a in [0.05, 0.10, 0.15, 0.20]:
        eurm = eurm_nlp * (1.0 - a) + eurm_top * a
        rec_list = eurm_to_recommendation_list(eurm, datareader=dr)
        ev.evaluate(rec_list, name='pop_first_k=' + str(k) + '_a=' + str(a))
