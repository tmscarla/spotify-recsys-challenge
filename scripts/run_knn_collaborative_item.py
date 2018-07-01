from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import eurm_to_recommendation_list
from recommenders.knn_collaborative_item import Knn_collaborative_item
import recommenders.similarity.similarity as sm
import scipy.sparse as sps
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
    name = "CFitem"
    knn = 100
    topk = 500

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        name = sys.argv[2]
        knn = int(sys.argv[3])
        topk = int(sys.argv[4])

    complete_name = mode+"_"+name+"_knn="+str(knn)+"_topk="+str(topk)

    if mode == "offline":

        """Test Set"""
        #Data initialization
        dr = Datareader(verbose=True, mode=mode, only_load=True)

        #Evaluetor initialization
        ev = Evaluator(dr)

        #Recommender algorithm initialization
        rec = Knn_collaborative_item()

        #Getting for the recommender algorithm
        urm = dr.get_urm()
        pid = dr.get_test_pids()

        #Fitting data
        rec.fit(urm, pid)

        #Computing similarity/model
        rec.compute_model(top_k= knn, sm_type=sm.COSINE ,shrink=200, alpha=0.1, beta=1, binary=True, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=topk,verbose=True, small=True)

        #evaluation and saving
        sps.save_npz(complete_name+".npz", rec.eurm)
        ev.evaluate(recommendation_list=eurm_to_recommendation_list(rec.eurm),name=name)

    if mode == "online":

        """Submission"""
        #Data initialization
        dr = Datareader(verbose=True, mode=mode, only_load=False)

        #Recommender algorithm initialization
        rec = Knn_collaborative_item()

        #Submitter initialization
        sb = Submitter(dr)

        #Getting for the recommender algorithm
        urm = dr.get_urm()
        pid = dr.get_test_pids()

        #Fitting data
        rec.fit(urm, pid)

        #Computing similarity/model
        rec.compute_model(top_k=knn, sm_type=sm.TVERSKY,shrink=200, alpha=0.1, beta=1, binary=True, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=topk, verbose=True, small=True)

        #submission
        sps.save_npz(complete_name+".npz", rec.eurm)
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(rec.eurm), name=name, track="main", verify=True, gzipped=False)




