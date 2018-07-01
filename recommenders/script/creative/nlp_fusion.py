from recommenders.similarity.s_plus import dot_product, tversky_similarity
from utils.post_processing import *
from utils.pre_processing import *
from scipy import sparse
from recommenders.nlp_strict import NLPStrict
from recommenders.nlp import NLP
from utils.sparse import *
import sys

arg = sys.argv[1:]
mode = arg[0]
if len(arg) > 1:
    topk = int(arg[1])
else:
    topk = 750

# INITIALIZATION
dr = Datareader(mode=mode, verbose=False, only_load=True)
test_pids = dr.get_test_pids()
urm = dr.get_urm()
urm.data = np.ones(len(urm.data))

# PARAMS
norm = True
work = True
split = True
skip_words = True
date = False
porter = False
porter2 = True
lanca = False
lanca2 = True
data1 = False

# NLP STRICT
nlp_strict = NLPStrict(dr)
ucm_strict = nlp_strict.get_UCM().astype(np.float64)
top_pop = dr.get_eurm_top_pop()

# Do not train on challenge set
ucm_strict_T = ucm_strict.copy()
inplace_set_rows_zero(ucm_strict_T, test_pids)
ucm_strict_T = ucm_strict_T.T

sim = tversky_similarity(ucm_strict, ucm_strict_T, k=450, alpha=0.2, beta=0.5,
                         shrink=150, target_items=test_pids)

# Compute eurm
eurm = dot_product(sim, urm, k=topk)
eurm = eurm.tocsr()
eurm = eurm[test_pids, :]

# NLP TOKENS
nlp = NLP(dr)

ucm = nlp.get_UCM(data1=data1).astype(np.float64)

# Do not train on challenge set
ucm_T = ucm.copy()
inplace_set_rows_zero(ucm_T, test_pids)
ucm_T = ucm_T.T

sim_lele = tversky_similarity(ucm, ucm_T, k=200, alpha=0.9, beta=1.0,
                              shrink=0, target_items=test_pids)

# Compute eurm
eurm_lele = dot_product(sim_lele, urm, k=topk)
eurm_lele = eurm_lele.tocsr()
eurm_lele = eurm_lele[test_pids, :]

# NLP FUSION
a = 0.2
eurm_l1 = norm_l1_row(eurm)
eurm_lele_l1 = norm_l1_row(eurm_lele)
nlp_fusion = a * eurm_l1 + (1.0 - a) * eurm_lele_l1

if dr.online():
    sparse.save_npz(ROOT_DIR + '/recommenders/script/creative/online_npz/nlp_fusion_online.npz', nlp_fusion)
else:
    sparse.save_npz(ROOT_DIR + '/recommenders/script/creative/offline_npz/nlp_fusion_offline.npz', nlp_fusion)

