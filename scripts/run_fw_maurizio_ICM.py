from personal.MaurizioFramework.CollaborativeFeatureWeighting.FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from recommenders.knn_content_item import Knn_content_item
from recommenders.similarity.similarity import *
import scipy.sparse as sps
from utils.definitions import ROOT_DIR
from utils.post_processing import eurm_to_recommendation_list,eurm_remove_seed
import scipy.sparse as sps




mode = "offline"
knn = 100
topk = 750

complete_name ="maurizio_"+ mode +"__knn=" + str(knn) + "_topk=" + str(topk)


if __name__ == '__main__':

    sim = sps.load_npz(ROOT_DIR+"/similarities/offline-similarity_rp3beta_knn100.npz")

    dr = Datareader(mode=mode, only_load=True)


    ######### MAURIZ
    ICM = dr.get_icm(alid=True)

    cfw = CFW_D_Similarity_Linalg(URM_train= dr.get_urm(),
                                  ICM= ICM.copy(),
                                  S_matrix_target= sim ,
                                  URM_validation = None)

    cfw.fit()

    weights = sps.diags(cfw.D_best)

    sps.save_npz("ICM_fw_maurizio", weights)

    ICM_weighted = ICM.dot(weights)

    sps.save_npz("ICM_fw_maurizio", ICM_weighted)


    ######## NOI
    urm = dr.get_urm()
    pid = dr.get_test_pids()

    cbfi = Knn_content_item()
    cbfi.fit(urm, ICM_weighted, pid)

    cbfi.compute_model(top_k=knn, sm_type=COSINE, shrink=0,  binary=False, verbose=True)
    cbfi.compute_rating(top_k=topk, verbose=True, small=True)

    sps.save_npz(complete_name+".npz", cbfi.eurm)
    ev = Evaluator(dr)
    ev.evaluate(recommendation_list=eurm_to_recommendation_list(cbfi.eurm),
                name=complete_name)



