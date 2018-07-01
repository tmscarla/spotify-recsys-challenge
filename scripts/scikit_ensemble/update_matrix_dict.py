from utils.definitions import *

print("[ Offline Dictionary ]")
matrix_dict = {"nlp": "nlp_fusion_tuned_offline.npz",
               "nlp_rp3beta": "empty.npz",
               "rp3beta2": "offline-rp3beta.npz",
               "cbf_user_artist": "eurm_cbfu_artists_offline.npz",
               "cbf_user_album": "eurm_cbfuser_albums_offline.npz",
               "pers_top_pop": "p_top_pop_offline.npz",
               "pers_top_pop_2": "top_pop_2_album_offline.npz",
               "slim": "slim_bpr_completo_test1.npz",
               "rp3beta_cat9": "offline_CAT9rp3b_cut=25_knn=850_topk=750_shrink250_2018-05-28_15-19_LAST.npz",
               "cbf_item_artist": "cb_ar_offline.npz",
               "cbf_item_album": "cb_al_offline.npz",
               "cbf_item_album_artist": "cb_al_ar_offline.npz",
               "cf_user": "cf_ub_offline.npz",
               "rp3beta": "cf_ib_offline.npz",
               "top_pop": "offline-top_pop-mean=28-perc=0.1.npz",
               "svd": "svd_offline.npz",
               "asl": "als_offline.npz"
               }

save_obj(matrix_dict, "matrix_dict", path="")

print("[ Online Dictionary ]")
matrix_dict_online  = {"nlp": "nlp_fusion_tuned_online.npz",
                       "nlp_rp3beta": "empty.npz",
                       "rp3beta2": "online-rp3beta",
                       "cbf_user_artist": "eurm_cbu_artists_online.npz",
                       "cbf_user_album": "eurm_cbf_user_albums_online.npz",
                       "pers_top_pop": "p_top_pop_offline.npz",
                       "pers_top_pop_2": "top_pop_2_album_offline.npz",
                       "slim": "slim_online.npz",
                       "rp3beta_cat9": "online_CAT9rp3b_cut=25_knn=850_topk=750_2018.npz",
                       "cbf_item_artist": "cb_ar_online.npz",
                       "cbf_item_album": "cb_al_online.npz",
                       "cbf_item_album_artist": "cb_al_ar_online.npz",
                       "cf_user": "cf_ub_online.npz",
                       "rp3beta": "cf_ib_online.npz",
                       "top_pop": "online-top_pop-mean=28-perc=0.1.npz"
                       }

save_obj(matrix_dict_online, "matrix_dict_online", path="")


