import logging

import scipy.sparse as sps

from boosts.hole_boost import HoleBoost
from boosts.tail_boost import TailBoost
from utils.datareader import Datareader
from utils.definitions import ROOT_DIR
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from utils.pre_processing import *

logging.basicConfig(filename='result.log',level=logging.DEBUG)

dr = Datareader(verbose=False, mode="offline", only_load=True)
ev = Evaluator(dr)

sim = sps.load_npz(ROOT_DIR + "/data/sim_offline.npz")

# rp3b = sps.load_npz(ROOT_DIR + "/data/sub/EURM-rp3beta-online.npz")
# knn_c_i_al = sps.load_npz(ROOT_DIR + "/data/sub/KNN CONTENT ITEM-album-top_k=850-sm_type=cosine-shrink=100.npz")
# knn_c_i_ar = sps.load_npz(ROOT_DIR + "/data/sub/KNN CONTENT ITEM-artist-top_k=850-sm_type=cosine-shrink=100.npz")
nlp = sps.load_npz(ROOT_DIR + "/data/eurm_nlp_offline.npz")
# cf_u = sps.load_npz(ROOT_DIR + "/data/sub/eurm_cfu_online.npz")

eurm_ens = sps.load_npz(ROOT_DIR + "/data/ENSEMBLED.npz")

#matrix = [rp3b, knn_c_i_ar, knn_c_i_al, nlp, cf_u]

#eurm_ens = ensembler(matrix, [0.720, 0.113, 0.177, 0.194, 1.0], normalization_type="max")


# HOLEBOOST
hb = HoleBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l1_row)
eurm_ens = hb.boost_eurm(categories=[8, 10], k=300, gamma=5)

# NINEBOOST
nb = TailBoost(similarity=sim, eurm=eurm_ens, datareader=dr, norm=norm_l2_row)
eurm_ens = nb.boost_eurm(last_tracks=10, k=100, gamma=0.01)

rec_list = eurm_to_recommendation_list(eurm_ens)
rec_list_nlp = eurm_to_recommendation_list(nlp)

indices = dr.get_test_pids_indices(cat=1)
for i in indices:
    rec_list[i] = rec_list_nlp[i]

# EVALUATION
ev.evaluate(rec_list, name='ens_with_cfu_nineboosted', show_plot=False)
