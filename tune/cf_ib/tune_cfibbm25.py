from recommenders.similarity.dot_product import dot_product_similarity, dot_product
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.post_processing import eurm_to_recommendation_list
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import recommenders.similarity.s_plus as ss
import numpy as np
from tqdm import tqdm
import utils.pre_processing as pre
import utils.sparse as ut
import utils.post_processing as post
from recommenders.cf_ib_bm25 import CF_IB_BM25

dr = Datareader(mode='offline', only_load=True, verbose=False)
urm = dr.get_urm()
pids = dr.get_test_pids()
rec = CF_IB_BM25(urm, binary=True, datareader=dr, mode='offline', verbose=True, verbose_evaluation= False)
rec.tune_alpha_beta(k=400, verbose_tune= True, save_mean=True, overwrite=True, range_alpha=np.arange(0.4,1.1,0.1))
print('DONE')