import sys
from scipy import sparse
import numpy as np
import utils.pre_processing as pre
from utils.definitions import *
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import *
from utils.post_processing import *
from fast_import import *

dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)
urm = dr.get_urm()

rec = CF_IB_BM25(urm=urm, datareader=dr, verbose_evaluation=False)
rec.model(alpha=1, beta=0, k=250)
rec.recommend(target_pids=None)

eurm_cf_i = rec.eurm

rec = CF_UB_BM25(urm=urm, datareader=dr, verbose_evaluation=False)
rec.model(alpha=1, beta=0.1, k=250)
rec.recommend(target_pids=None)

eurm_cf_u = rec.eurm

rows = []
cols = []
data = []

topk = 50

for idx in tqdm(range(eurm_cf_i.shape[0]), desc='URM augmented'):
    # Compute rows
    start_cfi = eurm_cf_i.indptr[idx]
    end_cfi = eurm_cf_i.indptr[idx+1]
    start_cfu = eurm_cf_u.indptr[idx]
    end_cfu = eurm_cf_u.indptr[idx+1]

    # Keep top
    top_cfi = np.argsort(eurm_cf_i.data[start_cfi:start_cfi])[::-1][:topk]
    top_cfu = np.argsort(eurm_cf_u.data[start_cfu:start_cfu])[::-1][:topk]

    top_tracks_cfi = eurm_cf_i.indices[top_cfi]
    top_tracks_cfu = eurm_cf_u.indices[top_cfu]

    intersect = np.intersect1d(top_tracks_cfi, top_tracks_cfu)

    for t in intersect:
        rows.append(idx)
        cols.append(t)
        data.append(1)

urm_derived = sparse.csr_matrix((data, (rows, cols)), shape=urm.shape)

urm_new = urm + urm_derived
urm_new.data = np.ones(len(urm_new.data))


#### RECOMMENDATION

rec = CF_UB_BM25(urm=urm_new, datareader=dr, verbose_evaluation=False)
rec.model(alpha=1, beta=0, k=250)
rec.urm = urm
rec.fast_recommend()
res = rec.fast_evaluate_eurm()
print(res[1])

