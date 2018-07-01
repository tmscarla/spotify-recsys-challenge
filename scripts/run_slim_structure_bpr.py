from personal.MaurizioFramework.SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet
from personal.MaurizioFramework.SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet_Cython
from personal.MaurizioFramework.SLIM_ElasticNet.SLIM_ElasticNet import MultiThreadSLIM_ElasticNet
from personal.MaurizioFramework.SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_BPR_Cython
from personal.MaurizioFramework.SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_MSE_Cython
from recommenders.similarity.dot_product import dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.bot import Bot_v1
from utils.post_processing import eurm_to_recommendation_list_submission


from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import sys

sys.stdout=open('output_da_salvare.txt','w')

def evaluate_shrinked(W_sparse, urm_shrinked,  pids_shrinked ):

    user_profile_batch = urm_shrinked[pids_shrinked]

    eurm = dot_product(user_profile_batch, W_sparse, k=500).tocsr()
    recommendation_list = np.zeros((10000, 500))
    for row in tqdm(range(eurm.shape[0]), desc="spotify rec list shrinked"):
        val = eurm[row].data
        ind = val.argsort()[-500:][::-1]
        ind = eurm[row].indices[ind]
        recommendation_list[row, 0:len(ind)] = ind

    ev.evaluate(recommendation_list=recommendation_list,
                name="slim_structure_parametribase_BPR_epoca_0_noepoche",
                return_overall_mean=False,
                show_plot=False, do_plot=True)

if __name__ == '__main__':

    l1 = 0.0001
    l2 = 0.0001
    topk = 150

    bot = Bot_v1("keplero slim_structure")
    try:
        ######################SHRINKED
        # dr = Datareader(mode="offline", train_format="400k", only_load=True)
        # ev = Evaluator(dr)
        # pids = dr.get_test_pids()
        #
        # urm, dictns, dict2 = dr.get_urm_shrinked()
        # urm_evaluation = dr.get_evaluation_urm()[pids]
        #
        # pids_converted = np.array([dictns[x] for x in pids], dtype=np.int32)

        ######### FULL
        dr = Datareader(mode="offline", only_load=True, verbose=False)
        ev = Evaluator(dr)
        pids = dr.get_test_pids()
        urm = dr.get_urm()

        slim = SLIM_Structure_BPR_Cython(urm)


        slim = SLIM_Structure_BPR_Cython(urm)
        slim.fit(epochs=1, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
                batch_size = 1, lambda_1 = l1, lambda_2 = l2, learning_rate = 1e-3, topK = topk,
                sample_quota = None, force_positive = False,
                sgd_mode='adam', gamma=0.995, beta_1=0.9, beta_2=0.999,
                stop_on_validation = False, lower_validatons_allowed = 3, validation_metric = "prec_t",
                validation_function = None, validation_every_n = 1
                )

        evaluate_shrinked(W_sparse= slim.W_sparse, urm_shrinked= urm, pids_shrinked= pids_converted)

        sps.save_npz("W_sparse_slim_structure.npz",slim.W_sparse)

    except Exception as e:
        bot.error("Exception " + str(e))

    bot.end()
