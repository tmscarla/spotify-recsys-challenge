from utils import post_processing as post
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.post_processing import eurm_remove_seed

mode = "online"

dr = Datareader(verbose=False, mode = mode, only_load="False")

name = mode+"/slim_online"


eurm = eurm_remove_seed(sps.load_npz(mode+"/slim_online.npz"), dr)
# sps.save_npz(mode+"/online_nlp_knn100_bm25.npz",eurm)



for i in range(1, 11):

    indices = dr.get_test_pids_indices(cat=i)
    save = eurm[indices]
    sps.save_npz(name+"-cat"+str(i)+".npz", save)
