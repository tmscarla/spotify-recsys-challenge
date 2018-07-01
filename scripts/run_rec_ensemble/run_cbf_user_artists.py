from utils.post_processing import eurm_to_recommendation_list_submission
from utils.post_processing import eurm_to_recommendation_list
from recommenders.knn_content_item import Knn_content_item
from recommenders.similarity.similarity import *
from recommenders.similarity.s_plus import *
from utils.evaluator import Evaluator
from utils.submitter import Submitter
from utils.datareader import Datareader
from utils.pre_processing import *
import utils.pre_processing as pre
import scipy.sparse as sps
from utils.sparse import *
import sys


if __name__ == '__main__':

    # SELECT EXECUTION MODE
    mode = "online"
    name = "cbf_user_artists"
    knn = 800
    topk = 750
    save_eurm = True
    complete_name = mode + "_" + name + "_knn=" + str(knn) + "_topk=" + str(topk)

    if mode == "offline":
        # Initialization
        dr = Datareader(verbose=False, mode=mode, only_load=True)
        test_pids = list(dr.get_test_pids())
        ev = Evaluator(dr)
        urm = dr.get_urm()

        # UCM
        ucm_artists = dr.get_ucm_artists()
        ucm_artists = bm25_row(ucm_artists)

        # Similarity
        print('Similarity..')
        sim = tversky_similarity(ucm_artists, ucm_artists.T, shrink=200, target_items=test_pids,
                                 alpha=0.1, beta=1, k=knn, verbose=1, binary=False)
        sim = sim.tocsr()

        # Prediction
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        # Save eurm
        if save_eurm:
            sps.save_npz('eurm_' + name + '_' + mode + '.npz', eurm)

        # Evaluation
        ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm, datareader=dr),
                    name=complete_name)

    elif mode == "online":
        # Initialization
        dr = Datareader(verbose=False, mode=mode, only_load=True)
        test_pids = list(dr.get_test_pids())
        sb = Submitter(dr)
        urm = dr.get_urm()

        # UCM
        ucm_artists = dr.get_ucm_albums()
        ucm_artists = bm25_row(ucm_artists)

        # Do not train on challenge set
        ucm_artists_T = ucm_artists.copy()
        inplace_set_rows_zero(ucm_artists_T, test_pids)
        ucm_artists_T = ucm_artists_T.T

        # Similarity
        print('Similarity..')
        sim = tversky_similarity(ucm_artists, ucm_artists_T, shrink=200, target_items=test_pids,
                                 alpha=0.1, beta=1, k=knn, verbose=1, binary=False)
        sim = sim.tocsr()

        # Prediction
        eurm = dot_product(sim, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_pids, :]

        # Save eurm
        if save_eurm:
            sps.save_npz('eurm_' + name + '_' + mode + '.npz', eurm)

        # Submission
        sb.submit(recommendation_list=eurm_to_recommendation_list(eurm, datareader=dr),
                  name=complete_name)

    else:
        print("Invalid mode!")

