"""
OFFLINE> python last_songs.py offline 25 100 500
"""

from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list
from recommenders.r_p_3_beta import R_p_3_beta
from utils.pre_processing import norm_max_row
from utils.post_processing import eurm_remove_seed
from utils.definitions import ROOT_DIR
import numpy as np
import scipy.sparse as sps
import datetime
from sklearn.preprocessing import normalize
import sys




if __name__ == '__main__':

    mode = "offline"
    cut = 25
    knn = 100
    topk = 750

    complete_name = 'cb_ib_cat9_'+mode+'.npz'

    if mode=="offline":

        #Data initialization
        dr = Datareader(verbose=False, mode=mode, only_load=True)

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

        # INJECTING URM POS with only last 25 songs
        rec.urm = dr.get_last_n_songs_urm(n=cut)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)
        lastsongs_eurm = rec.eurm.copy()

        sps.save_npz(complete_name, rec.eurm)

        ev = Evaluator(dr)
        ev.evaluate(eurm_to_recommendation_list(rec.eurm), 'prova', verbose=True)

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

        #Computing similarity/model
        rec.compute_model(top_k= knn, shrink=100, alpha=0.5, beta=0.5, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)
        normal_eurm = rec.eurm.copy()

        # INJECTING URM POS with only last 25 songs
        rec.urm = dr.get_last_n_songs_urm(n=cut)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)
        lastsongs_eurm = rec.eurm.copy()

        sps.save_npz(complete_name, rec.eurm)

        rec.eurm = eurm_remove_seed(rec.eurm,dr)

        #submission
        sb.submit(recommendation_list=eurm_to_recommendation_list(rec.eurm), name=name, track="main", verify=True, gzipped=False)









