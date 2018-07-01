from personal.MaurizioFramework.ParameterTuning.BayesianSearch import BayesianSearch
from personal.MaurizioFramework.ParameterTuning.AbstractClassSearch import DictionaryKeys
from utils.definitions import ROOT_DIR
import pickle
from personal.MaurizioFramework.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recommenders.similarity.dot_product import dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.bot import Bot_v1
from tqdm import tqdm
import scipy.sparse as sps
import numpy as np
import sys



def run_SLIM_bananesyan_search(URM_train, URM_validation, logFilePath = ROOT_DIR+"/results/logs_baysian/"):

    recommender_class = SLIM_BPR_Cython
    bananesyan_search = BayesianSearch(recommender_class, URM_validation=URM_validation,
                                        evaluation_function=evaluateRecommendationsSpotify_BAYSIAN)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [100, 150, 200, 250, 300, 350, 400, 500]
    hyperparamethers_range_dictionary["lambda_i"] = [1e-7,1e-6,1e-5,1e-4,1e-3,0.001,0.01,0.05,0.1]
    hyperparamethers_range_dictionary["lambda_j"] = [1e-7,1e-6,1e-5,1e-4,1e-3,0.001,0.01,0.05,0.1]
    hyperparamethers_range_dictionary["learning_rate"] = [0.1,0.01,0.001,0.0001,0.00005,0.000001, 0.0000001]
    hyperparamethers_range_dictionary["minRatingsPerUser"] = [0,  5, 50, 100]


    logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch Results.txt", "a")

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS:  {
                                                                        "URM_train":URM_train,
                                                                        "positive_threshold":0,
                                                                        "URM_validation":URM_validation,
                                                                        "final_model_sparse_weights":True,
                                                                        "train_with_sparse_weights":True,
                                                                        "symmetric" : True},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {
                                                                "epochs" : 5,
                                                                "beta_1" : 0.9,
                                                                "beta_2" : 0.999,
                                                                "validation_function": evaluateRecommendationsSpotify_RECOMMENDER,
                                                                "stop_on_validation":True ,
                                                                "sgd_mode" : 'adam',
                                                                "validation_metric" : "ndcg_t",
                                                                "lower_validatons_allowed":3,
                                                                "validation_every_n":1},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    best_parameters = bananesyan_search.search(recommenderDictionary,
                                               metric="ndcg_t",
                                               n_cases=200,
                                               output_root_path=""+logFilePath + recommender_class.RECOMMENDER_NAME,
                                               parallelPoolSize=4)

    logFile.write("best_parameters: {}".format(best_parameters))
    logFile.flush()
    logFile.close()

    pickle.dump(best_parameters, open(logFilePath + recommender_class.RECOMMENDER_NAME + "_best_parameters", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)


def evaluateRecommendationsSpotify_RECOMMENDER(recommender):
    """
    THIS FUNCTION WORKS INSIDE THE RECOMMENDER
    :param self:
    :return:
    """
    user_profile_batch = recommender.URM_train[pids_converted]

    eurm = dot_product(user_profile_batch, recommender.W_sparse, k=500).tocsr()
    recommendation_list = np.zeros((10000, 500))
    for row in tqdm(range(eurm.shape[0]), desc="spotify rec list"):
        val = eurm[row].data
        ind = val.argsort()[-500:][::-1]
        ind = eurm[row].indices[ind]
        recommendation_list[row, 0:len(ind)] = ind

    prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a = ev.evaluate(recommendation_list=recommendation_list,
                                                                name=recommender.configuration+"epoca"+
                                                                     str(recommender.currentEpoch),
                                                                return_overall_mean=True, verbose = False,
                                                                show_plot=False, do_plot=True)
    results_run = {}
    results_run["prec_t"] = prec_t
    results_run["ndcg_t"] = ndcg_t
    results_run["clicks_t"] = clicks_t
    results_run["prec_a"] = prec_a
    results_run["ndcg_a"] = ndcg_a
    results_run["clicks_a"] = clicks_a
    return (results_run)

def evaluateRecommendationsSpotify_BAYSIAN(recommender, URM_validation, paramether_dictionary) :
    """
    THIS FUNCTION WORKS INSIDE THE BAYSIAN-GRID SEARCH
    :param self:
    :return:
    """
    user_profile_batch = recommender.URM_train[pids_converted]
    eurm = dot_product(user_profile_batch, recommender.W_sparse, k=500).tocsr()

    recommendation_list = np.zeros((10000, 500))
    for row in tqdm(range(eurm.shape[0]), desc="spotify rec list"):
        val = eurm[row].data
        ind = val.argsort()[-500:][::-1]
        ind = eurm[row].indices[ind]
        recommendation_list[row, 0:len(ind)] = ind
    prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a = ev.evaluate(recommendation_list=recommendation_list,
                                                                name=recommender.configuration+"epoca"+str(recommender.currentEpoch),
                                                                return_overall_mean=True, verbose= False,
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

    bot = Bot_v1("keplero bananesyan slim")

    try:
        ######################SHRINKED
        dr = Datareader(mode="offline", train_format="100k", only_load=True)
        ev = Evaluator(dr)
        pids = dr.get_test_pids()

        urm, dictns, dict2 = dr.get_urm_shrinked()
        urm_evaluation = dr.get_evaluation_urm()[pids]

        pids_converted = np.array([dictns[x] for x in pids], dtype=np.int32)

        run_SLIM_bananesyan_search(URM_train=urm, URM_validation=urm_evaluation)

        # dr = Datareader(mode="offline",  only_load=True)
        # ev = Evaluator(dr)
        # pids = dr.get_test_pids()
        #
        # urm = dr.get_urm()
        # urm_evaluation = dr.get_evaluation_urm()[pids]
        # pids_converted = pids
        #
        # run_SLIM_bananesyan_search(URM_train=urm, URM_validation=urm_evaluation)


    except Exception as e:
        bot.error("Exception "+str(e))

    bot.end()