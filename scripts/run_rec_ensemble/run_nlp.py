from utils.submitter import Submitter
from utils.post_processing import eurm_to_recommendation_list_submission
from recommenders.nlp import NLP
import sys
import datetime
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.evaluator import Evaluator
import numpy as np
from recommenders.similarity.s_plus import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from utils.post_processing import eurm_to_recommendation_list, eurm_remove_seed
from utils.pre_processing import bm25_row
from utils.sparse import *


if __name__ == '__main__':

    mode = "online"
    name = "nlp"
    knn = 200
    topk = 600
    alpha= 0.8
    beta = 1.0
    save_eurm = True


    dr = Datareader(verbose=True, mode=mode,only_load=True)

    complete_name = mode+"_"+name+"_knn="+str(knn)+"_topk="+str(topk)\
                    + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")



    if mode == 'offline':
        # Setup
        urm = dr.get_urm()
        test_pids = dr.get_test_pids()

        # Init object
        nlp = NLP(dr)

        # Get ucm
        ucm = nlp.get_UCM()


        # Compute similarity (playlists x playlists)
        sim = tversky_similarity(ucm, ucm.T, k=knn, shrink=0, alpha=1, beta=0.1)
        sim = sim.tocsr()

        # Recommendation
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        rec_list = eurm_to_recommendation_list(eurm, dr)

        if save_eurm:
            sps.save_npz(mode + "_" + name + ".npz", eurm, compressed=False)

        # Submission
        ev = Evaluator(dr)
        ev.evaluate(rec_list, name=name)

    elif mode == 'online':
        # Setup
        sb = Submitter(dr)
        urm = dr.get_urm()
        test_pids = dr.get_test_pids()

        # Init object
        nlp_strict = NLP(dr)

        # Get ucm
        ucm = nlp_strict.get_UCM()
        print(ucm.shape)

        # Do not train on challenge set
        ucm_T = ucm.copy()
        inplace_set_rows_zero(ucm_T, test_pids).astype(np.float64)
        ucm_T = ucm_T.T

        # Compute similarity (playlists x playlists)
        sim = tversky_similarity(ucm, ucm_T, shrink=200, alpha=0.9, beta=1, k=knn)
        sim = sim.tocsr()

        # Recommendation
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        rec_list = eurm_to_recommendation_list(eurm, datareader=dr)

        if save_eurm:
            sps.save_npz(mode + "_" + name + ".npz", eurm, compressed=False)

        # Submission
        sb.submit(rec_list, name=name)

    else:
        print('Wrong mode!')
