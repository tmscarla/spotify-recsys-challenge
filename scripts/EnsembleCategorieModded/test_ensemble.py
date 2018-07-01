import scipy.sparse as sps
from utils.pre_processing import norm_max_row

import sys
from utils.evaluator import Evaluator
from utils.pretty_printer import Pretty_printer
from utils.datareader import Datareader
from utils.post_processing import eurm_to_recommendation_list_submission
from utils.ensembler import ensembler
import numpy as np
from tqdm import tqdm
from utils.post_processing import  eurm_to_recommendation_list
from utils.submitter import Submitter


if __name__ == '__main__':



    res = []
    mode = "offline"

    dr = Datareader(verbose=False, mode = mode, only_load="False")
    if mode=="offline":
        ev = Evaluator(dr)

    w = []
    for i in range(1, 11):
        arg = np.load("cat" + str(i) + "/best.npy")
        print(arg)
        best = list(arg[1:].astype(np.float))
        w.append(best)

    for i in tqdm(range(1,11)):
        if mode == "offline":

            CBF_ALBUM = sps.load_npz(mode+"/offline-cbf_item_album-cat"+str(i)+".npz")
            CBF_ARTISTA = sps.load_npz(mode+"/offline-cbf_item_artist-cat"+str(i)+".npz")
            NLP = norm_max_row(sps.load_npz(mode + "/nlp_eurm_offline_bm25-cat" + str(1) + ".npz"))
            RP3BETA = sps.load_npz(mode+"/offline-rp3beta-cat"+str(i)+".npz")
            CF_USER = sps.load_npz(mode + "/cfu_eurm-cat"+str(i)+".npz")
            SLIM = sps.load_npz(mode +"/slim_bpr_completo_test1-cat"+str(i)+".npz")
            CBF_USER_ARTIST = sps.load_npz(mode +"/eurm_cbfu_artists_offline-cat"+str(i)+".npz")


        matrix = [CBF_ALBUM, CBF_ARTISTA, NLP, RP3BETA, CF_USER, SLIM, CBF_USER_ARTIST]

        we = w[i-1]

        res.append(ensembler(matrix, we, normalization_type="lele"))

    ret = sps.vstack(res).tocsr()
    if mode == "offline":
        ev.evaluate(eurm_to_recommendation_list(ret), "best_test", verbose=True)

#    sps.save_npz("ensemble_per_cat_"+mode+"_new_data_28_maggio.npz", ret)
    if mode == "online":
        sb = Submitter(dr)
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(ret), name="best_test", track="main", verify=True, gzipped=False)
