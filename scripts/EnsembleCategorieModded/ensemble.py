import scipy.sparse as sps
import numpy as np
from utils.ensembler import ensembler
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import  eurm_to_recommendation_list
from utils.post_processing import  eurm_remove_seed
from utils.submitter import Submitter
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.definitions import *


mode = "offline"
type = "unique"



w = []
print("[ Loading weights ]")
for i in range(1, 11):
    arg = np.load("weight/cat" + str(i) + ".npy")
    print(arg[- 1])
    w.append(list(arg[:len(arg) - 1].astype(np.float)))

print("[ Loading matrix name ]")
if mode == "offline":
    matrix_dict = load_obj("matrix_dict", path="")
    dir = "offline/"

if mode == "online":
    matrix_dict = load_obj("matrix_dict_online", path="")
    dir = "online/"

if type == "unique":
    print("[ Loading cat 1 ]")
    cat = 1
    a = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["top_pop_cat1"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp_rp3beta"]+"-cat" + str(cat) + ".npz")
    matrix = [[a, b, c]]

    print("[ Loading cat 2 ]")
    cat = 2
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["pers_top_pop"]+"-cat" + str(cat) + ".npz")
    i = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    l = sps.load_npz(dir+matrix_dict["pers_top_pop_2"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h, i, l])

    print("[ Loading cat 3 ]")
    cat = 3
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")

    matrix.append([a, b, c, d, e, f, g, h])

    print("[ Loading cat 4 ]")
    cat = 4
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g])

    print("[ Loading cat 5 ]")
    cat = 5
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h])

    print("[ Loading cat 6 ]")
    cat = 6
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")

    matrix.append([a, b, c, d, e, f, g])

    print("[ Loading cat 7 ]")
    cat = 7
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h])

    print("[ Loading cat 8 ]")
    cat = 8
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h])

    print("[ Loading cat 9 ]")
    cat = 9
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["rp3beta_cat9"]+"-cat" + str(cat) + ".npz")
    i = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h, i])

    print("[ Loading cat 10 ]")
    cat = 10
    a = sps.load_npz(dir+matrix_dict["cbf_item_album"]+"-cat" + str(cat) + ".npz")
    b = sps.load_npz(dir+matrix_dict["cbf_item_artist"]+"-cat" + str(cat) + ".npz")
    c = sps.load_npz(dir+matrix_dict["nlp"]+"-cat" + str(cat) + ".npz")
    d = sps.load_npz(dir+matrix_dict["rp3beta"]+"-cat" + str(cat) + ".npz")
    e = sps.load_npz(dir+matrix_dict["cf_user"]+"-cat" + str(cat) + ".npz")
    f = sps.load_npz(dir+matrix_dict["slim"]+"-cat" + str(cat) + ".npz")
    g = sps.load_npz(dir+matrix_dict["cbf_user_artist"]+"-cat" + str(cat) + ".npz")
    h = sps.load_npz(dir+matrix_dict["cbf_user_album"]+"-cat" + str(cat) + ".npz")
    matrix.append([a, b, c, d, e, f, g, h])

    rprec = []
    for i in range(0, 10):
        print("[ Ensembling cat", i+1, "]")
        rprec.append(ensembler(matrix[i], w[i], normalization_type="max"))
    res = sps.vstack(rprec).tocsr()

    import time
    name = "ensemble-"+mode+"-data-"+time.strftime("%x")+"-"+time.strftime("%X")
    name = name.replace("/", "_")
    sps.save_npz("results/"+name+".npz", res)

    print("[ Initizalizing Datereader ]")
    dr = Datareader(verbose=False, mode=mode, only_load="False")




    res = eurm_to_recommendation_list(res, datareader=dr)

    if mode == "offline":
        print("[ Initizalizing Evaluator ]")
        ev = Evaluator(dr)
        ev.evaluate(res, name="ens")

    if mode == "online":
        print("[ Initizalizing Submitter ]")
        sb = Submitter(dr)
        sb.submit(recommendation_list=res, name=name, track="main", verify=True,
                  gzipped=False)


#
#
#
# if type == "splitted":
#     mode = "offline"
#
#     print("[ Loading weights ]")
#     w_rprec = []
#     tmp = 0
#     for i in range(1, 11):
#         arg = np.load("rprec/cat" + str(i) + ".npy")
#         tmp += -float(arg[-1])
#         w_rprec.append(list(arg[:len(arg)-1].astype(np.float)))
#
#     w_ndcg = []
#     tmp = 0
#     for i in range(1, 11):
#         arg = np.load("rprec/cat" + str(i) + ".npy")
#         tmp += -float(arg[-1])
#         w_ndcg.append(list(arg[:len(arg)-1].astype(np.float)))
#
#
#     if mode == "offline":
#         cat = 1
#         print("[ Loading cat 1 ]")
#         a = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-top_pop-mean=28-perc=0.1-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/offline-nlp_into_rp3beta-cat" + str(cat) + ".npz")
#
#
#         matrix = [[a, b, c]]
#
#
#         print("[ Loading cat 2 ]")
#         cat = 2
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/p_top_pop_offline-cat" + str(cat) + ".npz")
#         i = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#         l = sps.load_npz("offline/top_pop_2_album_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h, i ,l])
#
#
#
#         print("[ Loading cat 3 ]")
#         cat = 3
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 4 ]")
#         cat = 4
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#
#
#         matrix.append([a, b, c, d, e, f, g])
#
#
#         print("[ Loading cat 5 ]")
#         cat = 5
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 6 ]")
#         cat = 6
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g])
#
#
#         print("[ Loading cat 7 ]")
#         cat = 7
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 8 ]")
#         cat = 8
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 9 ]")
#         cat = 9
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/offline_CAT9rp3b_cut=25_knn=850_topk=750_shrink250_2018-05-28_15-19_LAST-cat" + str(cat) + ".npz")
#         i = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h, i])
#
#
#         print("[ Loading cat 10 ]")
#         cat = 10
#
#         a = sps.load_npz("offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#         rprec = []
#         for i in range(0, 10):
#             print("[ RPrec - Ensembling cat", i, "]")
#             rprec.append(ensembler(matrix[i], w_rprec[i], normalization_type="max"))
#         rprec = sps.vstack(rprec).tocsr()
#
#
#         sps.save_npz("rprec_offline.npz", rprec)
#
#     if mode == "offline":
#
#
#
#         cat = 1
#         print("[ Loading cat 1 ]")
#         a = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-top_pop-mean=28-perc=0.1-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/offline-nlp_into_rp3beta-cat" + str(cat) + ".npz")
#
#
#         matrix = [[a, b, c]]
#
#
#         print("[ Loading cat 2 ]")
#         cat = 2
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/p_top_pop_offline-cat" + str(cat) + ".npz")
#         i = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#         l = sps.load_npz("ndcg_offline/top_pop_2_album_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h, i ,l])
#
#
#
#         print("[ Loading cat 3 ]")
#         cat = 3
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 4 ]")
#         cat = 4
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#
#
#         matrix.append([a, b, c, d, e, f, g])
#
#
#         print("[ Loading cat 5 ]")
#         cat = 5
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 6 ]")
#         cat = 6
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g])
#
#
#         print("[ Loading cat 7 ]")
#         cat = 7
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 8 ]")
#         cat = 8
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#
#         print("[ Loading cat 9 ]")
#         cat = 9
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/offline_CAT9rp3b_cut=25_knn=850_topk=750_shrink250_2018-05-28_15-19_LAST-cat" + str(cat) + ".npz")
#         i = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#         matrix.append([a, b, c, d, e, f, g, h, i])
#
#
#         print("[ Loading cat 10 ]")
#         cat = 10
#
#         a = sps.load_npz("ndcg_offline/offline-cbf_item_album-cat" + str(cat) + ".npz")
#         b = sps.load_npz("ndcg_offline/offline-cbf_item_artist-cat" + str(cat) + ".npz")
#         c = sps.load_npz("ndcg_offline/eurm_nlp_fusion_offline-cat" + str(cat) + ".npz")
#         d = sps.load_npz("ndcg_offline/offline-rp3beta-cat" + str(cat) + ".npz")
#         e = sps.load_npz("ndcg_offline/offline-cfuser-cat" + str(cat) + ".npz")
#         f = sps.load_npz("ndcg_offline/slim_bpr_completo_test1-cat" + str(cat) + ".npz")
#         g = sps.load_npz("ndcg_offline/eurm_cbfu_artists_offline-cat" + str(cat) + ".npz")
#         h = sps.load_npz("ndcg_offline/eurm_cbfuser_albums_offline-cat" + str(cat) + ".npz")
#
#
#
#         matrix.append([a, b, c, d, e, f, g, h])
#
#         ndcg = []
#         for i in range(0, 10):
#             print("[ RPrec - Ensembling cat", i, "]")
#             ndcg.append(ensembler(matrix[i], w_ndcg[i], normalization_type="max"))
#         ndcg = sps.vstack(ndcg).tocsr()
#
#         es = []
#         sps.save_npz("ndcg_offline.npz", ndcg)
#
#
#         print("[ Initizalizing Datereader ]")
#         dr = Datareader(verbose=False, mode=mode, only_load="False")
#
#
#
#         from utils.post_processing import append_rec_list
#         from utils.post_processing import remove_rec_list_after_cutoff
#         res = append_rec_list(remove_rec_list_after_cutoff(eurm_to_recommendation_list(rprec, datareader=dr), dr), eurm_to_recommendation_list(ndcg, datareader=dr))
#
#
#         if mode == "offline":
#             print("[ Initizalizing Evaluator ]")
#             ev = Evaluator(dr)
#             ev.evaluate(res, name="ens")
#
#         # sps.save_npz("ensemble_per_cat_" + mode + "test_blocchi.npz", res)
#
#
#
#
#
#     # if mode == "online":
#     #     print("[ Loading cat 1 ]")
#     #     a = sps.load_npz("online/online_nlp_knn100_bm25-cat1.npz")
#     #     b = sps.load_npz("online/online-top_pop-mean=28-perc=0.1.npz")[0:1000]
#     #
#     #     matrix = [[a, b]]
#     #
#     #
#     #     print("[ Loading cat 2 ]")
#     #     cat = 2
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat"+str(cat)+".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat"+str(cat)+".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat"+str(cat)+".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat"+str(cat)+".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat"+str(cat)+".npz")
#     #     f = sps.load_npz("online/slim_online-cat"+str(cat)+".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat"+str(cat)+".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#     #
#     #
#     #
#     #     print("[ Loading cat 3 ]")
#     #     cat = 3
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#     #
#     #
#     #     print("[ Loading cat 4 ]")
#     #     cat = 4
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f])
#     #
#     #
#     #     print("[ Loading cat 5 ]")
#     #     cat = 5
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#     #
#     #
#     #     print("[ Loading cat 6 ]")
#     #     cat = 6
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f])
#     #
#     #
#     #     print("[ Loading cat 7 ]")
#     #     cat = 7
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#     #
#     #
#     #     print("[ Loading cat 8 ]")
#     #     cat = 8
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#     #
#     #
#     #     print("[ Loading cat 9 ]")
#     #     cat = 9
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #     h = sps.load_npz("online/online_CAT9rp3b_cut=25_knn=850_topk=750_2018-05-29_11-17_LAST.npz")[8000:9000]
#     #
#     #     matrix.append([a, b, c, d, e, f, g, h])
#     #
#     #
#     #     print("[ Loading cat 10 ]")
#     #     cat = 10
#     #
#     #     a = sps.load_npz("online/online-cbf_item-album-850-cat" + str(cat) + ".npz")
#     #     b = sps.load_npz("online/online-cbf_item-artist-850-cat" + str(cat) + ".npz")
#     #     c = sps.load_npz("online/online_nlp_knn100_bm25-cat" + str(cat) + ".npz")
#     #     d = sps.load_npz("online/online-rp3beta-cat" + str(cat) + ".npz")
#     #     e = sps.load_npz("online/online-cf_user-850-cat" + str(cat) + ".npz")
#     #     f = sps.load_npz("online/slim_online-cat" + str(cat) + ".npz")
#     #     g = sps.load_npz("online/eurm_cbu_artists_online-cat" + str(cat) + ".npz")
#     #
#     #
#     #
#     #
#     #     matrix.append([a, b, c, d, e, f, g])
#
#     #
#     #
#     # res = []
#     # for i in range(0, 10):
#     #     print("[ Ensembling cat",i,"]")
#     #     res.append(ensembler(matrix[i], w[i], normalization_type="max"))
#     #
#     # print("[ Initizalizing Datereader ]")
#     # dr = Datareader(verbose=False, mode=mode, only_load="False")
#     #
#     # ret = eurm_remove_seed(sps.vstack(res).tocsr(), datareader=dr)
#     #
#     # if mode == "offline":
#     #     print("[ Initizalizing Evaluator ]")
#     #     ev = Evaluator(dr)
#     #     from utils.post_processing import shift_rec_list_cutoff
#     #
#     #     ev.evaluate(shift_rec_list_cutoff(eurm_to_recommendation_list(ret), dr), name="ens")
#     #
#     # sps.save_npz("ensemble_per_cat_" + mode + "4_giugno.npz", ret)
#     #
#     # if mode == "online":
#     #     sb = Submitter(dr)
#     #     sb.submit(recommendation_list=eurm_to_recommendation_list_submission(eurm_remove_seed(ret, dr)), name="best_test", track="main",
#     #               verify=True, gzipped=False)
