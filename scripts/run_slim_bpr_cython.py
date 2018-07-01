from personal.MaurizioFramework.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recommenders.similarity.dot_product import dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.submitter import Submitter
from utils.post_processing import eurm_remove_seed
from utils.post_processing import eurm_to_recommendation_list
from utils.post_processing import eurm_to_recommendation_list_submission
from tqdm import tqdm
import sys
import numpy as np
import scipy.sparse as sps
from utils.definitions import ROOT_DIR

from utils.bot import Bot_v1

def evaluate_for_online(self):
    results_run = {}
    results_run["prec_t"] = 1
    results_run["ndcg_t"] = 1
    results_run["clicks_t"] = 1
    results_run["prec_a"] = 1
    results_run["ndcg_a"] = 1
    results_run["clicks_a"] = 1
    return (results_run)


def evaluateRecommendationsSpotify(self):
    # print("Recommender: sparsity self.W_sparse:", self.W_sparse.nnz / self.W_sparse.shape[1] / self.W_sparse.shape[0])

    user_profile_batch = self.URM_train[pids_converted]
    print("dot product")
    eurm = dot_product(user_profile_batch, self.W_sparse, k=750).tocsr()
    eurm = eurm_remove_seed(eurm)


    recommendation_list = np.zeros((10000, 500))
    for row in range(eurm.shape[0]):
        val = eurm[row].data
        ind = val.argsort()[-500:][::-1]
        ind = eurm[row].indices[ind]
        recommendation_list[row, 0:len(ind)] = ind

    prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a = ev.evaluate(recommendation_list=recommendation_list,
                                                                name=self.configuration+"_epoca"+str(self.currentEpoch),
                                                                return_overall_mean=True, verbose=False,
                                                                show_plot=False, do_plot=True)

    results_run = {}
    results_run["prec_t"] = prec_t
    results_run["ndcg_t"] = ndcg_t
    results_run["clicks_t"] = clicks_t
    results_run["prec_a"] = prec_a
    results_run["ndcg_a"] = ndcg_a
    results_run["clicks_a"] = clicks_a

    return (results_run)


if __name__ == '__main__':

    mode = "offline"
    name = "slim"
    epochs = 5
    min_rating = 0
    lambda_i = 0.001
    lambda_j = 0.000001
    learning_rate = 0.001
    topk = 300
    beta_1 = 0.9
    beta_2 = 0.999

    train_format = ''  # only if offline '50k' '100k' ...

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        name = sys.argv[2]
        epochs = int(sys.argv[3])
        min_rating = int(sys.argv[4])
        lambda_i = int(sys.argv[5])
        lambda_j = int(sys.argv[6])
        learning_rate = int(sys.argv[7])
        topk = int(sys.argv[8])
        beta_1 = int(sys.argv[9])
        beta_2 = int(sys.argv[10])

    complete_name = mode + "_" + name + "_epochs=" + str(epochs) +"_minR=" + str(min_rating) \
                    + "_li=" + str(lambda_i) + "_lj=" + str(lambda_j) + \
                    "_lr=" + str(learning_rate) + "_topk=" + str(topk) + \
                    "_b1=" + str(beta_1) + "_b2=" + str(beta_2)

    if len(sys.argv) == 12:
        assert mode == "offline"
        train_format = sys.argv[11]
        complete_name += "_shrink=" + train_format

    bot = Bot_v1("keplero  slim " + mode)

    dr = Datareader(mode=mode, verbose=True, train_format=train_format, only_load=True)

    if mode == 'offline':

        if len(train_format>0):
            ####### DATA INIZIALIZATION SHRINKED #################
            dr = Datareader(mode=mode, train_format=train_format, only_load=True)

            ev = Evaluator(dr)
            pids = dr.get_test_pids()
            urm, dict_n_to_s, dict2= dr.get_urm_shrinked()
            urm_evaluation = dr.get_evaluation_urm()
            urm_evaluation = urm_evaluation[pids]
            pids_converted = np.array([dict_n_to_s[x] for x in pids], dtype=np.int32)


        else:
            ####### DATA INIZIALIZATION FULL #################
            dr = Datareader(mode=mode, only_load=True, verbose=False)
            ev = Evaluator(dr)
            pids = dr.get_test_pids()

            urm = dr.get_urm()
            urm_evaluation = dr.get_evaluation_urm()
            urm_evaluation = urm_evaluation[pids]
            urm_evaluation = None
            pids_converted = pids

        slim = SLIM_BPR_Cython(URM_train=urm, positive_threshold=0, URM_validation=urm_evaluation,
                               final_model_sparse_weights=True, train_with_sparse_weights=True,
                               symmetric=True)

        slim.fit(epochs=1, logFile=None, filterTopPop=False, minRatingsPerUser=min_rating,
                 batch_size=1000, lambda_i=lambda_i, lambda_j=lambda_j, learning_rate=learning_rate, topK=topk,
                 sgd_mode='adam', gamma=0.999, beta_1=beta_1, beta_2=beta_2,
                 stop_on_validation=True, lower_validatons_allowed=1, validation_metric="ndcg_t",
                 validation_function=evaluate_for_online, validation_every_n=1)

        # calculating eurm, evaluation, save
        user_profile_batch = slim.URM_train[pids_converted]
        eurm = dot_product(user_profile_batch, slim.W_sparse, k=500).tocsr()
        recommendation_list = eurm_to_recommendation_list(eurm)

        sps.save_npz(ROOT_DIR+"/results/"+complete_name+".npz", eurm, compressed=False)
        ev.evaluate(recommendation_list=recommendation_list, name=complete_name)




    elif mode =="online":
        ####### DATA INIZIALIZATION ONLINE #################
        dummy_variable = 0
        dr = Datareader(mode="online", only_load=True, verbose=False)
        pids = dr.get_test_pids()

        urm= dr.get_urm()
        urm_evaluation = None
        pids_converted = pids

        slim = SLIM_BPR_Cython(URM_train=urm, positive_threshold=0, URM_validation=urm_evaluation,
                               final_model_sparse_weights=True, train_with_sparse_weights=True,
                               symmetric=True)

        slim.fit(epochs=1, logFile=None, filterTopPop=False, minRatingsPerUser=min_rating,
                 batch_size=1000, lambda_i=lambda_i, lambda_j=lambda_j, learning_rate=learning_rate, topK=topk,
                 sgd_mode='adam', gamma=0.999, beta_1=beta_1, beta_2=beta_2,
                 stop_on_validation=True, lower_validatons_allowed=1, validation_metric="ndcg_t",
                 validation_function=evaluate_for_online, validation_every_n=1)

        user_profile_batch = slim.URM_train[pids_converted]
        eurm = dot_product(user_profile_batch, slim.W_sparse, k=500).tocsr()
        recommendation_list = eurm_to_recommendation_list(eurm)

        # calculating eurm, evaluation, save
        user_profile_batch = slim.URM_train[pids_converted]
        eurm = dot_product(user_profile_batch, slim.W_sparse, k=500).tocsr()
        recommendation_list = eurm_to_recommendation_list(eurm)

        sps.save_npz(ROOT_DIR + "/results/" + complete_name + ".npz", eurm, compressed=False)

        sb = Submitter(dr)
        sb.submit(recommendation_list=eurm_to_recommendation_list_submission(eurm),
                  name=name, track="main", verify=True, gzipped=False)



    else:
        print("invalid mode.")






    # ev.evaluate(recommendation_list=recommendation_list,
    #              name="slim ")


    # except Exception as e:
    #     bot.error("Exception "+str(e))
    #
    # bot.end()

