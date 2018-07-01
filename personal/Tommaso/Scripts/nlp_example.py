import numpy as np
from personal.Tommaso.NLP.GA_FeatureSelection import GA_FeatureSelection
from personal.Tommaso.NLP.NLP import NLP
from utils.datareader import Datareader
from utils.definitions import *
from utils.evaluator import Evaluator
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
import time
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.tversky import tversky_similarity
from utils.post_processing import *
from utils.pre_processing import *
from scipy import sparse


def online():
    datareader = Datareader(mode='online', only_load=True)

    print('NLP...')
    stopwords = STOP_WORDS
    token_weights = np.array(TOKEN_WEIGHTS)

    nlp = NLP(datareader, stopwords=[])
    ucm = nlp.get_ucm()
    #ucm = bm25_row(ucm)
    #inplace_csr_column_scale(ucm, token_weights)

    urm = datareader.get_urm_shrinked()[0]

    print('Computing similarity...')
    start = time.time()
    # Compute similarity
    similarity = tversky_similarity(ucm, shrink=200, alpha=0.1, beta=1)
    similarity = similarity.tocsr()
    print(time.time() - start)

    print('Computing eurm...')
    start = time.time()
    # Compute eurm
    eurm_nlp = dot_product(similarity, urm, k=500)
    eurm_nlp = eurm_nlp.tocsr()
    print(eurm_nlp.shape)
    eurm_nlp = eurm_nlp[-10000:, :]

    sparse.save_npz(ROOT_DIR + '/data/eurm_nlp_no_stop_online.npz', eurm_nlp)


def new():
    datareader = Datareader(mode='offline', only_load=True)
    evaluator = Evaluator(datareader)

    print('NLP...')
    stopwords = STOP_WORDS
    token_weights = np.array(TOKEN_WEIGHTS)
    test_playlists = datareader.get_test_pids()

    nlp = NLP(datareader=datareader, stopwords=[], mode='both')
    print('Getting ucm and icm...')
    ucm = nlp.get_ucm()
    ucm = bm25_row(ucm)
    icm = nlp.get_icm()
    icm = bm25_row(icm)
    icm_T = icm.T

    #ucm = bm25_row(ucm)

    #urm = datareader.get_urm()

    print('Computing eurm...')
    start = time.time()
    eurm_nlp = dot_product(ucm[test_playlists, :], icm_T, k=500)
    print(time.time() - start)

    print('Converting to csr...')
    eurm_nlp = eurm_nlp.tocsr()
    print(eurm_nlp.shape)
    #eurm_nlp = eurm_nlp[test_playlists:, :]

    sparse.save_npz(ROOT_DIR + '/data/eurm_nlp_new_method_offline.npz', eurm_nlp)
    evaluator.evaluate(eurm_to_recommendation_list(eurm_nlp), name='nlp_new_method', show_plot=False)

def icm():
    datareader = Datareader(mode='offline', only_load=True)
    evaluator = Evaluator(datareader)

    print('NLP...')
    stopwords = STOP_WORDS
    token_weights = np.array(TOKEN_WEIGHTS)
    test_playlists = datareader.get_test_pids()

    nlp = NLP(datareader=datareader, stopwords=[], mode='tracks')
    print('Getting ucm and icm...')
    icm = nlp.get_icm()
    icm = bm25_row(icm)

    print('Computing similarity...')
    start = time.time()
    # Compute similarity
    similarity = tversky_similarity(icm, shrink=200, alpha=0.1, beta=1)
    similarity = similarity.tocsr()
    print(time.time() - start)

    urm = datareader.get_urm()

    print('Computing eurm...')
    start = time.time()
    # Compute eurm
    eurm_nlp = dot_product(urm[test_playlists, :], similarity, k=500)
    eurm_nlp = eurm_nlp.tocsr()

    # sparse.save_npz(ROOT_DIR + '/data/eurm_nlp_weighted_offline.npz', eurm_nlp)
    evaluator.evaluate(eurm_to_recommendation_list(eurm_nlp), name='nlp_enriched')


def prova():

    dr = Datareader(mode='offline', only_load=True)
    print(dr.get_artist_to_tracks_dict())
    exit()

    dr = Datareader(mode='offline', only_load=True, verbose=False)
    test_playlists = dr.get_test_pids()

    stopwords = STOP_WORDS
    token_weights = np.array(TOKEN_WEIGHTS)

    nlp = NLP(mode='playlists', datareader=dr, stopwords=STOP_WORDS)
    s = nlp.get_ucm()
    print(s.shape)

    evaluator = Evaluator(dr)

    ucm = nlp.get_ucm()
    sim = sparse.load_npz(ROOT_DIR + '/data/cf_user_similarity.npz')

    print('Computing dot...')
    ucm = dot_product(sim, ucm, k=200)
    print('NNZ', ucm.nnz)
    exit()

    urm = dr.get_urm()

    # ucm = ucm.astype(np.float64)
    # inplace_csr_column_scale(ucm, token_weights)

    print('Computing similarity...')
    start = time.time()
    # Compute similarity
    similarity = tversky_similarity(ucm, shrink=200, alpha=0.1, beta=1)
    similarity = similarity.tocsr()
    print(time.time() - start)

    print('Computing eurm...')
    start = time.time()
    # Compute eurm
    eurm_nlp = dot_product(similarity, urm, k=500)
    eurm_nlp = eurm_nlp.tocsr()
    eurm_nlp = eurm_nlp[test_playlists, :]

    #sparse.save_npz(ROOT_DIR + '/data/eurm_nlp_weighted_offline.npz', eurm_nlp)
    evaluator.evaluate(eurm_to_recommendation_list(eurm_nlp), name='nlp_enriched')


