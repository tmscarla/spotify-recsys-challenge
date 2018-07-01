import scipy.sparse as sps
import sys
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.post_processing import eurm_to_recommendation_list
from utils.ensembler import ensembler
import numpy as np

import os.path

dr = Datareader(verbose=False, mode = "offline", only_load="False")
cat = 9

a = sps.load_npz("../offline/offline-cbf_item_album-cat"+str(cat)+".npz")
b = sps.load_npz("../offline/offline-cbf_item_artist-cat"+str(cat)+".npz")
c = sps.load_npz("../offline/nlp_eurm_offline_bm25-cat"+str(cat)+".npz")
d = sps.load_npz("../offline/offline-rp3beta-cat"+str(cat)+".npz")
e = sps.load_npz("../offline/offline-cfuser-cat"+str(cat)+".npz")
f = sps.load_npz("../offline/slim_bpr_completo_test1-cat"+str(cat)+".npz")
g = sps.load_npz("../offline/eurm_cbfu_artists_offline-cat"+str(cat)+".npz")
h = sps.load_npz("../offline/offline_CAT9rp3b_cut=25_knn=850_topk=750_shrink250_2018-05-28_15-19_LAST.npz")[8000:9000]

matrix = [a, b, c, d, e, f, g, h]

a = float(sys.argv[1])
b = float(sys.argv[2])
c = float(sys.argv[3])
d = float(sys.argv[4])
e = float(sys.argv[5])
f = float(sys.argv[6])
g = float(sys.argv[7])
h = float(sys.argv[8])


res = ensembler(matrix, [a, b, c, d, e, f, g, h], normalization_type="max")

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