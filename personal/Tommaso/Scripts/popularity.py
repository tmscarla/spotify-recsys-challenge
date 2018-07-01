import sys
from scipy import sparse
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from utils.ensembler import *
from recommenders.similarity.s_plus import *
from utils.evaluator import Evaluator


dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)
test_pids = dr.get_test_pids()

n_clusters = 1000

urm = dr.get_urm()

#######

icm = dr.get_icm(alid=True, arid=True)
icm = bm25_col(icm)
#######

# icm = dr.get_icm_popularity(n_clusters)
# icm = bm25_col(icm)
#
# icm_std = dr.get_icm(alid=True)
#
# icm_fusion = sp.hstack((icm, icm_std))
# icm_fusion = bm25_col(icm_fusion)

print('Similarity..')
# sim = tversky_similarity(icm, icm.T, shrink=200, alpha=0.1,
#                          beta=1, k=200, verbose=1, binary=False)
sim = dot_product(icm, icm.T, k=70)
sim = sim.tocsr()

sim_rp3 = sparse.load_npz(ROOT_DIR + '/data/sim_offline.npz')

for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    newsim = norm_l1_row(sim) * a + norm_l1_row(sim_rp3) * (1-a)

    # Prediction
    eurm = dot_product(urm, newsim, k=750)
    eurm = eurm.tocsr()
    eurm = eurm[test_pids, :]

    rec_list = eurm_to_recommendation_list(eurm, datareader=dr)

    ev.evaluate(rec_list, name='popprova')
