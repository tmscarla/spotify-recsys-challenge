from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import  eurm_to_recommendation_list
from recommenders.r_p_3_beta import R_p_3_beta
import recommenders.similarity.similarity as sm
import numpy as np
from utils import ensembler
import utils.sparse as ut
import scipy.sparse as sps
import gc
from sklearn.preprocessing import normalize
import sys


'''
This file contains just an example on how to run the algorithm.
The parameter used are just the result of a first research of the optimum value.
To run this file just set the parameter at the start of the main function or set from console as argv parameter.
As argv you can even set mode of execution (online, offline) and the name of the result file
'''

if __name__ == '__main__':

    ### Select execution mode: 'offline', 'online' ###
    mode = "offline"
    name = "rp3beta_depopularized_250"
    if (len(sys.argv) > 1):
        mode = sys.argv[1]
        name = sys.argv[2]

    if mode == "offline":

        #Data initialization
        dr = Datareader(verbose=False, mode='offline', only_load=True)

        #Evaluetor initialization
        ev = Evaluator(dr)

        #Recommender algorithm initialization
        rec = R_p_3_beta()

        #Getting for the recommender algorithm
        urm = dr.get_urm()
        pids = dr.get_test_pids()
        urm.data = np.ones(len(urm.data))

        # Depopularized
        top = urm.sum(axis=0).A1
        mask = np.argsort(top)[::-1][:250]
        ut.inplace_set_cols_zero(urm, mask)

        p_ui = normalize(urm, norm="l1")
        p_iu = normalize(urm.T, norm="l1")

        #Fitting data
        rec.fit(p_ui, p_iu, top, pids)

        #Computing similarity/model
        rec.compute_model(top_k=800, shrink=100, alpha=0.5, beta=0, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=750,verbose=True, small=True)

        sps.save_npz(mode+"-"+name+".npz", rec.eurm)

        #evaluation
        ev.evaluate(eurm_to_recommendation_list(rec.eurm), name, verbose=True)

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
        rec.compute_model(top_k=100, shrink=100, alpha=0.5, beta=0.4, verbose=True)

        # Computing ratings
        rec.compute_rating(top_k=500, verbose=True, small=True)

        sps.save_npz(mode+"-"+name+".npz", rec.eurm)

        #submission
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(rec.eurm), name=name, track="main", verify=True, gzipped=False)





