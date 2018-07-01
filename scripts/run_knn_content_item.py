from utils.post_processing import eurm_to_recommendation_list_submission
from utils.post_processing import eurm_to_recommendation_list
from recommenders.knn_content_item import Knn_content_item
from recommenders.similarity.similarity import *
from utils.evaluator import Evaluator
from utils.submitter import Submitter
from utils.datareader import Datareader
import utils.pre_processing as pre
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
    mode = "online"
    name = "content_item"
    feature_type = "artist"
    knn = 100
    topk = 500

    album = True
    artist = False

    if len(sys.argv) > 1:

        mode = sys.argv[1]
        name = sys.argv[2]
        feature_type = sys.argv[3] #artist or album
        knn = int(sys.argv[4])
        topk = int(sys.argv[5])

        if feature_type == 'album':
            album = True
            artist = False

        elif feature_type == 'artist':
            album = False
            artist = True
        else:
            print("invalid type")


    complete_name = mode+"_"+name+"_"+feature_type+"_knn="+str(knn)+"_topk="+str(topk)


    if mode == "offline":
        dr = Datareader(verbose=False, mode=mode, only_load=True)
        urm = dr.get_urm()
        icm = dr.get_icm(arid=artist, alid=album)
        pid = dr.get_test_pids()
        icm_bm25 = pre.bm25_row(icm)

        cbfi = Knn_content_item()
        cbfi.fit(urm, icm_bm25, pid)

        cbfi.compute_model(top_k=knn, sm_type=TVERSKY, shrink=100, alpha=0.1, binary=False, verbose=True)
        cbfi.compute_rating(top_k=topk, verbose=True, small=True)

        sps.save_npz(complete_name+".npz", cbfi.eurm)
        ev = Evaluator(dr)
        ev.evaluate(recommendation_list=eurm_to_recommendation_list(cbfi.eurm),
                    name=complete_name)

    elif mode == "online":
        dr = Datareader(verbose=False, mode=mode, only_load=True)
        urm = dr.get_urm()
        icm = dr.get_icm(arid=artist, alid=album)
        pid = dr.get_test_pids()
        icm_bm25 = pre.bm25_row(icm)

        cbfi = Knn_content_item()
        cbfi.fit(urm, icm_bm25, pid)

        cbfi.compute_model(top_k=knn, sm_type=TVERSKY, shrink=100, alpha=0.1, binary=False, verbose=True)
        cbfi.compute_rating(top_k=topk, verbose=True, small=True)

        sps.save_npz(complete_name+".npz", cbfi.eurm)
        sb = Submitter(dr)
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(cbfi.eurm),
                  name=complete_name ,
                  track="main", verify=True, gzipped=False)

    else:
        print("invalid mode.")

