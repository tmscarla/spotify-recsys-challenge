from scipy import sparse

import utils.pre_processing as pre
from boosts.hole_boost import HoleBoost
from utils.datareader import Datareader
from utils.definitions import ROOT_DIR
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list

# Initialization
dr = Datareader(mode='offline', only_load=True)
ev = Evaluator(dr)

# Load matrices
eurm = sparse.load_npz(ROOT_DIR + '/data/eurm_rp3_offline.npz')
sim = sparse.load_npz(ROOT_DIR + '/data/sim_offline.npz')
print('Loaded')

# Normalization
eurm = pre.norm_l2_row(eurm)
sim = pre.norm_l2_row(sim)


# HoleBoost

h = HoleBoost(sim, eurm, dr)
eurm_b = h.boost_eurm(categories=[2,3,4,5,6,7,8,9,10], k=200, gamma=10)

#sparse.save_npz(ROOT_DIR + '/data/eurm_boosted_online.npz', eurm_b)
rec_list = eurm_to_recommendation_list(eurm_b)

# Evaluation
ev.evaluate(rec_list, name='rp3_l2_all_200_10', save=True, show_plot=False)
