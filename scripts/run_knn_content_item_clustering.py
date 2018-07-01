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

import numpy as np

def clusterize_icm(icm, n_clusters = 1000):

    # Clusterizing means aggregating features with OR
    print("ICM shape is {}, using {} clusters".format(icm.shape, n_clusters))

    n_features = icm.shape[1]

    cluster_allocation = np.random.randint(0, n_clusters, n_features)

    # Use array as it requires MUCH less space than lists
    dataBlock = 10000000

    clustered_icm_data = np.zeros(dataBlock, dtype=np.float64)
    clustered_icm_row = np.zeros(dataBlock, dtype=np.int32)
    clustered_icm_col = np.zeros(dataBlock, dtype=np.int32)

    numCells = 0

    for cluster_index in range(n_clusters):

        cluster_allocation_mask = cluster_allocation==cluster_index

        new_feature_column = icm[:,cluster_allocation_mask].sum(axis=1)
        new_feature_column = np.array(new_feature_column).ravel()

        nonzero_items = new_feature_column.nonzero()[0]

        # clustered_icm_row.extend(nonzero_items.tolist())
        # clustered_icm_col.extend([cluster_index]*len(nonzero_items))
        # clustered_icm_data.extend(new_feature_column[nonzero_items])

        for el_index in range(len(nonzero_items)):
            if numCells == len(clustered_icm_row):
                clustered_icm_row = np.concatenate((clustered_icm_row, np.zeros(dataBlock, dtype=np.int32)))
                clustered_icm_col = np.concatenate((clustered_icm_col, np.zeros(dataBlock, dtype=np.int32)))
                clustered_icm_data = np.concatenate((clustered_icm_data, np.zeros(dataBlock, dtype=np.float64)))


            clustered_icm_row[numCells] = nonzero_items[el_index]
            clustered_icm_col[numCells] = cluster_index
            clustered_icm_data[numCells] = new_feature_column[nonzero_items[el_index]]

            numCells += 1



    new_shape = (icm.shape[0], n_clusters)

    clustered_icm = sps.csr_matrix((clustered_icm_data, (clustered_icm_row, clustered_icm_col)), shape=new_shape)

    print("Clustering done!")

    return clustered_icm, cluster_allocation




if __name__ == '__main__':

    ### Select execution mode: 'offline', 'online' ###
    mode = "offline"
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

        clustered_icm, cluster_allocation = clusterize_icm(icm, n_clusters = 1000)


        icm_bm25 = pre.bm25_row(clustered_icm)

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

