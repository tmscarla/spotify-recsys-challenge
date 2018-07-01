from personal.MaurizioFramework.CollaborativeFeatureWeighting.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from recommenders.knn_content_item import Knn_content_item
from recommenders.similarity.similarity import *
import scipy.sparse as sps
from recommenders.similarity.s_plus import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from utils.definitions import ROOT_DIR
from utils.post_processing import eurm_to_recommendation_list,eurm_remove_seed
import scipy.sparse as sps
from recommenders.nlp import NLP



mode = "offline"
knn = 100
topk = 750

complete_name ="maurizio_"+ mode +"__knn=" + str(knn) + "_topk=" + str(topk)


if __name__ == '__main__':

    sim_user = sps.load_npz(ROOT_DIR+"/similarities/cf_user_similarity.npz")

    dr = Datareader(mode=mode, only_load=True)

    ######### MAURIZ
    nlp = NLP(dr)
    UCM = nlp.get_UCM()

    cfw = CFW_D_Similarity_Linalg(URM_train= dr.get_urm().T,
                                  ICM= UCM.copy(),
                                  S_matrix_target= sim_user ,
                                  URM_validation = None)

    cfw.fit()


    weights = sps.diags(cfw.D_best)

    sps.save_npz("ucm_weights_maurizi", weights)

    UCM_weighted = dot_product(UCM,weights)

    sps.save_npz("ucm_fw_maurizio", UCM_weighted)

    ######## NOI
    urm = dr.get_urm()
    pid = dr.get_test_pids()

    similarity = tversky_similarity(UCM_weighted,UCM_weighted.T,
                                    binary=False, shrink=1, alpha=0.9, beta=1)
    similarity = similarity.tocsr()

    # eurm
    test_playlists = dr.get_test_pids()
    eurm = dot_product(similarity, urm, k=topk)
    eurm = eurm.tocsr()
    eurm = eurm[test_playlists, :]

    rec_list = eurm_to_recommendation_list(eurm)

    # evaluate
    ev = Evaluator(dr)
    ev.evaluate(rec_list, name='weighter', verbose=True, show_plot=False)
