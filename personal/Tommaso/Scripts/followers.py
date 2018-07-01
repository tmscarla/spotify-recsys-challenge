import sys
from scipy import sparse
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from utils.ensembler import *
from recommenders.similarity.s_plus import *
from utils.evaluator import Evaluator


# Initialization
dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)
test_pids = dr.get_test_pids()
urm = dr.get_urm()

# Parameters
n_clusters = 100
knn = 850
topk = 750

# UCMs
ucm_album = dr.get_ucm_albums(remove_duplicates=True)
#ucm = dr.get_ucm_followers(n_clusters)

#ucm = sparse.hstack((ucm_album, ucm_followers))
#ucm = bm25_row(ucm)

# Similarity
print('Similarity..')
sim = tversky_similarity(ucm_album, ucm_album.T, shrink=200, target_items=test_pids,
                         alpha=0.1, beta=1, k=knn, verbose=1, binary=False)
sim = sim.tocsr()

# Prediction
eurm = dot_product(sim, urm, k=topk)
eurm = eurm.tocsr()
eurm = eurm[test_pids, :]

# Evaluation
ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm, datareader=dr),
            name='ucm_album_followers')
