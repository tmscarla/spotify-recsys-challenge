"""
EXAMPLE:

ONLINE:     python run_nlp.py online 100 500
OFFLINE     python run_nlp.py offline 100 500 1

"""
from utils.submitter import Submitter
from utils.post_processing import eurm_to_recommendation_list_submission
from recommenders.nlp import NLP
import sys
import datetime
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.evaluator import Evaluator
import numpy as np
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.tversky import tversky_similarity
from utils.post_processing import eurm_to_recommendation_list, eurm_remove_seed
from utils.pre_processing import bm25_row


if __name__ == '__main__':

    mode = "offline"
    name = "nlp"
    knn = 100
    topk = 750

    test_num = "1"


    if len(sys.argv) > 1:

        mode = sys.argv[1]
        knn = int(sys.argv[2])
        topk = sys.argv[3]

        if mode=="offline":
            test_num = int(sys.argv[4])

    dr = Datareader(verbose=True, mode=mode,test_num=test_num, only_load=True)

    complete_name = mode+"_"+name+"_knn="+str(knn)+"_topk="+str(topk)\
                    + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if mode=="offline":
        complete_name+="_test="+str(test_num)

    # best: norm, wor, split, skipw, porter2, lanca2
    norm = True
    work = True
    split = True
    skip_words = True
    date = False
    porter = False
    porter2 = True
    lanca = False
    lanca2 = True
    data1 = False

    if mode == "offline":

        nlp = NLP(dr, stopwords=[], norm=norm, work=work, split=split, date=date, skip_words=skip_words,
                  porter=porter, porter2=porter2, lanca=lanca, lanca2=lanca2)

        ucm = nlp.get_UCM(data1=data1)
        urm = dr.get_urm()
        test_playlists = dr.get_test_pids()
        ucm= bm25_row(ucm)

        similarity = tversky_similarity(ucm, binary=False, shrink=1, alpha=0.1, beta=1)
        similarity = similarity.tocsr()

        #eurm
        eurm = dot_product(similarity, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[test_playlists, :]

        rec_list = eurm_to_recommendation_list(eurm)

        sps.save_npz(mode+"_"+name+"_bm25.npz", eurm, compressed=False)
        np.save(mode+"_"+name+"_bm25",rec_list)

        #evaluate
        ev = Evaluator(dr)
        ev.evaluate(rec_list, name=name, verbose=True, show_plot=False)


    if mode == "online":

        nlp = NLP(dr, stopwords=[], norm=norm, work=work, split=split, date=date, skip_words=skip_words,
                  porter=porter, porter2=porter2, lanca=lanca, lanca2=lanca2)

        pids = list(dr.get_train_pids()) + list(dr.get_test_pids())
        test_playlists = dr.get_test_pids()

        ucm = nlp.get_UCM(data1=data1)

        dr_old = Datareader(mode='online', only_load='True', type='old')

        train = ucm[:1000000]
        test = ucm[1000000:]

        test_indices = []
        for cat in range(1, 11):
            indices = dr_old.get_test_pids_indices(cat=cat)
            test_indices.extend(indices)

        new_indices = list(dr.get_train_pids()) + list(np.array(test_indices)+1000000)
        ucm = ucm[new_indices]





        urm = dr.get_urm()
        urm = urm[pids]

        ucm = bm25_row(ucm)

        similarity = tversky_similarity(ucm, binary=False, shrink=1, alpha=0.1, beta=1)
        similarity = similarity.tocsr()

        print(similarity.shape, urm.shape)
        eurm = dot_product(similarity, urm, k=topk)
        eurm = eurm.tocsr()
        eurm = eurm[-10000:]

        eurm = eurm_remove_seed(eurm, dr)
        rec_list = eurm_to_recommendation_list(eurm)


        sps.save_npz(mode + "_" + name + "_knn"+str(knn)+"_bm25.npz", eurm, compressed=False)
        np.save(mode + "_" + name + "_knn"+str(knn)+"_bm25", rec_list)

        sb = Submitter(dr)
        sb.submit(rec_list, name=name, track="main", verify=True, gzipped=False)



