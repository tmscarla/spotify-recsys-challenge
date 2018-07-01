from personal.MaurizioFramework.SLIM_ElasticNet.DSLIM_RMSE import  DSLIM_RMSE, MultiThreadDSLIM_RMSE
from recommenders.similarity.dot_product import dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.bot import Bot_v1
from utils.post_processing import eurm_remove_seed, eurm_to_recommendation_list

import datetime
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import sys


def evaluate_shrinked(W_sparse, urm_shrinked,  pids_shrinked ):

    W_sparse = W_sparse[pids_shrinked]

    eurm = dot_product(W_sparse, urm_shrinked, k=750).tocsr()

    eurm = eurm_remove_seed(eurm=eurm)

    rec_list = eurm_to_recommendation_list(eurm)


    ev.evaluate(recommendation_list=rec_list,
                name="slim_structure_parametribase_BPR_epoca_0_noepoche",
                return_overall_mean=False,
                show_plot=False, do_plot=True)


if __name__ == '__main__':

    mode = "offline"
    name = "DSLIM"
    l1 = 0.1
    l2 = 0.1
    beta = 0.2
    knn = 100
    topk = 750

    test_num = "1"


    if len(sys.argv) > 1:

        mode = sys.argv[1]
        l1 = int(sys.argv[2])
        l2 = int(sys.argv[3])
        beta = int(sys.argv[4])
        knn = int(sys.argv[5])
        topk = int(sys.argv[6])

        if mode=="offline":
            test_num = int(sys.argv[7])

    name ="DSLIM"
    complete_name = mode+"_"+name+"_knn="+str(knn)+"_topk="+str(topk)\
                    + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    if mode=="offline":
        complete_name+="_test="+str(test_num)


        bot = Bot_v1(complete_name)

        try:
            ######################SHRINKED
            dr = Datareader(mode=mode, test_num=test_num, train_format="50k", only_load=True)
            ev = Evaluator(dr)
            pids = dr.get_test_pids()

            urm, dictns, dict2 = dr.get_urm_shrinked()
            urm_evaluation = dr.get_evaluation_urm()[pids]

            pids_converted = np.array([dictns[x] for x in pids], dtype=np.int32)

            slim = MultiThreadDSLIM_RMSE(urm.T)

            slim.fit(l1_penalty=l1, l2_penalty=l2, positive_only=True, beta=beta, topK=topk)

            evaluate_shrinked(W_sparse= slim.W_sparse, urm_shrinked= urm, pids_shrinked= pids_converted)

            sps.save_npz(complete_name+".npz",slim.W_sparse,)

        except Exception as e:
            bot.error("Exception "+str(e))

        bot.end()

    else:

        print("online not implemented")