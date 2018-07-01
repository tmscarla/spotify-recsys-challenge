import scipy.sparse as sps
import sys
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.post_processing import eurm_to_recommendation_list
from utils.ensembler import ensembler
import numpy as np

import os.path

dr = Datareader(verbose=False, mode = "offline", only_load="False")
cat = 1

a = sps.load_npz("../offline/nlp_eurm_offline_bm25-cat"+str(cat)+".npz")
b = sps.load_npz("../../../personal/Sebastiano/top_pop-mean=28-perc=0.25.npz")[0:1000]
c = sps.load_npz("../offline/offline-nlp_into_rp3beta.npz").tocsr()[0:1000]

matrix = [a, b, c]

a = float(sys.argv[1])
b = float(sys.argv[2])
c = float(sys.argv[3])


res = ensembler(matrix, [a, b, c], normalization_type="max")

# from utils.post_processing import eurm_remove_seed
# res = eurm_remove_seed(res)


ev = Evaluator(dr)
ret = [-ev.evaluate_single_metric(eurm_to_recommendation_list(res, cat=cat), cat=cat, name="ens"+str(cat), metric='prec', level='track')]

if os.path.isfile("best.npy"):
    best = np.load("best.npy")
    if ret[0] < best[-1].astype(np.float):
        b = sys.argv[1:]
        b.append(ret[0])
        np.save("best", b)
else:
    b = sys.argv[1:]
    b.append(ret[0])
    np.save("best", b)

np.save("ret", ret)