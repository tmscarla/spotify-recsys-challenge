from utils.pre_processing import *
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.post_processing import *
from tqdm import tqdm
from scipy import sparse
import utils.sparse as ut
import pandas as pd
import numpy as np
import sys


datareader = Datareader(mode='offline', only_load=True, verbose=False)
evaluator = Evaluator(datareader)

urm = datareader.get_urm()
ucm_album = datareader.get_ucm_albums()

albums_pop = ucm_album.sum(axis=0).A1
mask = np.argsort(albums_pop)[::-1][:100]
ut.inplace_set_cols_zero(ucm_album, mask)

ucm_album = bm25_row(ucm_album)

print('Similarity..')
sim = tversky_similarity(ucm_album, ucm_album.T, shrink=200, alpha=0.1, beta=1, k=800, verbose=1, binary=False)
sim = sim.tocsr()

test_pids = list(datareader.get_test_pids())

eurm = dot_product(sim, urm, k=750)
eurm = eurm.tocsr()
eurm = eurm[test_pids, :]
sparse.save_npz('eurm_albums_depop_100_offline.npz', eurm)

eurm = eurm_remove_seed(eurm, datareader)

evaluator.evaluate(eurm_to_recommendation_list(eurm), name='cbuser_album_depop_100', show_plot=False)
