import numpy as np
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import norm_l1_row
from utils.post_processing import eurm_to_recommendation_list, eurm_remove_seed
from personal.Ervin.Word2Vec_recommender import W2VRecommender
from personal.Ervin.ItemRank import ItemRank
from personal.Ervin.tf_collaborative_user import TF_collaborative_user
from recommenders.knn_collaborative_item import Knn_collaborative_item


if __name__ == '__main__':
    dr = Datareader(only_load=True, mode='offline', test_num='1', verbose=False)
    pid = dr.get_test_playlists().transpose()[0]
    urm = dr.get_urm()
    urm.data = np.ones(len(urm.data))
    ev = Evaluator(dr)

    TFRec = Knn_collaborative_item()
    W2V = W2VRecommender()
    TFRec.fit(urm, pid)
    W2V.fit(urm, pid)

    TFRec.compute_model(verbose=True, top_k=850)
    TFRec.compute_rating(top_k=750, verbose=True, small=True)
    W2V.compute_model(verbose=True, size=50, window=None)
    W2V.compute_rating(verbose=True, small=True, top_k=750)
    TFRec.eurm = norm_l1_row(eurm_remove_seed(TFRec.eurm, dr))
    W2V.eurm = norm_l1_row(eurm_remove_seed(W2V.eurm, dr))

    for alpha in np.arange(0.9, 0, -0.1):
        print('[ Alpha = {:.1f}]'.format(alpha))
        eurm = alpha * TFRec.eurm + (1-alpha)*W2V.eurm
        ev.evaluate(recommendation_list=eurm_to_recommendation_list(eurm, remove_seed=False, datareader=dr),
                name="KNNItem_W2V"+str(alpha), old_mode=False, save=True)