from utils.submitter import Submitter
from utils.post_processing import eurm_to_recommendation_list_submission
from recommenders.nlp_strict import NLPStrict
import sys
import datetime
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.evaluator import Evaluator
import numpy as np
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from utils.post_processing import eurm_to_recommendation_list, eurm_remove_seed
from utils.pre_processing import bm25_row
from utils.sparse import *


if __name__ == '__main__':

    mode = "online"
    name = "nlp_strict"
    knn = 50
    topk = 750
    save_eurm = True

    if mode == 'offline':
        # Setup
        dr = Datareader(mode=mode, verbose=False, only_load=True)
        ev = Evaluator(dr)
        urm = dr.get_urm()
        test_pids = dr.get_test_pids()

        # Init object
        nlp_strict = NLPStrict(dr)

        # Get ucm
        ucm = nlp_strict.get_UCM()

        # Compute similarity (playlists x playlists)
        sim = tversky_similarity(ucm, ucm.T, shrink=200, alpha=0.1, beta=1, k=knn)
        sim = sim.tocsr()

        # Recommendation
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        rec_list = eurm_to_recommendation_list(eurm, dr)

        if save_eurm:
            sps.save_npz(mode + "_" + name + ".npz", eurm, compressed=False)

        # Submission
        ev.evaluate(rec_list, name=name)

    elif mode == 'online':
        # Setup
        dr = Datareader(mode=mode, verbose=False, only_load=True)
        sb = Submitter(dr)
        urm = dr.get_urm()
        test_pids = dr.get_test_pids()

        # Init object
        nlp_strict = NLPStrict(dr)

        # Get ucm
        ucm = nlp_strict.get_UCM()
        print(ucm.shape)

        # Do not train on challenge set
        ucm_T = ucm.copy()
        inplace_set_rows_zero(ucm_T, test_pids)
        ucm_T = ucm_T.T

        # Compute similarity (playlists x playlists)
        sim = tversky_similarity(ucm, ucm_T, shrink=200, alpha=0.1, beta=1, k=knn)
        sim = sim.tocsr()

        # Recommendation
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        rec_list = eurm_to_recommendation_list(eurm, dr)

        if save_eurm:
            sps.save_npz(mode + "_" + name + ".npz", eurm, compressed=False)

        # Submission
        sb.submit(rec_list, name=name)

    else:
        print('Wrong mode!')
