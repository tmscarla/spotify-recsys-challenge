"""
OFFLINE> python last_songs.py offline 25 100 500
"""

from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import  eurm_to_recommendation_list
from utils.pre_processing import norm_max_row
from recommenders.r_p_3_beta import R_p_3_beta
from utils.pre_processing import norm_max_row
import recommenders.similarity.similarity as sm
import numpy as np
from utils import ensembler
import scipy.sparse as sps
import datetime
import gc
from sklearn.preprocessing import normalize
import sys




if __name__ == '__main__':

    mode = "offline"
    name = "CAT9rp3b"
    cut = 25
    knn = 100
    topk = 500


    if (len(sys.argv) > 1):
        mode = sys.argv[1]
        cut = int(sys.argv[2])
        knn = int(sys.argv[3])
        topk = int(sys.argv[4])


    dr = Datareader(verbose=False, mode=mode, only_load=True)

    complete_name = mode+"_"+name+"_cut="+str(cut)+"_knn="+str(knn)+"_topk="+str(topk)\
                    + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if mode=="offline":

        #Data initialization
        dr = Datareader(verbose=False, mode='offline', only_load=True)

        #Recommender algorithm initialization
        rec = R_p_3_beta()

        #Getting for the recommender algorithm
        urm = dr.get_urm()
        pids = dr.get_test_pids()
        urm.data = np.ones(len(urm.data))
        p_ui = normalize(urm, norm="l1")
        p_iu = normalize(urm.T, norm="l1")
        top = urm.sum(axis=0).A1

        #Fitting data
        rec.fit(p_ui, p_iu, top, pids)

        #Computing similarity/model
        rec.compute_model(top_k= knn, shrink=100, alpha=0.5, beta=0.5, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)
        normal_eurm = rec.eurm.copy()

        sps.save_npz(complete_name+"_NORMAL.npz", rec.eurm)

        # INJECTING URM POS with only last 25 songs
        rec.urm = dr.get_last_n_songs_urm(n=cut)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)
        lastsongs_eurm = rec.eurm.copy()

        sps.save_npz(complete_name+"_LAST.npz", rec.eurm)

        eurm = norm_max_row(normal_eurm) + norm_max_row(lastsongs_eurm)

        sps.save_npz(complete_name+".npz", rec.eurm)

        #evaluation
        ev = Evaluator(dr)
        ev.evaluate(eurm_to_recommendation_list(rec.eurm), name, verbose=True)


#TODO
    if mode == "online":

        ### Submission ###

        #Data initialization
        dr = Datareader(verbose=True, mode='online', only_load=True)

        #Recommender algorithm initialization
        rec = R_p_3_beta()

        #Submitter initialization
        sb = Submitter(dr)

        #Getting data ready for the recommender algorithm
        urm = dr.get_urm()
        pids = dr.get_test_pids()
        urm.data = np.ones(len(urm.data))
        p_ui = normalize(urm, norm="l1")
        p_iu = normalize(urm.T, norm="l1")
        top = urm.sum(axis=0).A1

        # Fitting data
        rec.fit(p_ui, p_iu, top, pids)

        # Computing similarity/model
        rec.compute_model(top_k=knn, shrink=100, alpha=0.5, beta=0.4, verbose=True)

        # Computing ratings
        rec.compute_rating(top_k=topk, verbose=True, small=True)

        sps.save_npz(mode+"-"+name+".npz", rec.eurm)

        #submission
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(rec.eurm), name=name, track="main", verify=True, gzipped=False)









