import time
import pandas
import scipy.sparse as sps
from utils.datareader import Datareader
from utils.evaluator import Evaluator
import pandas as pd
import numpy as np
from personal.Lele.NLP import NLP2
from utils.definitions import ROOT_DIR
from utils.definitions import STOP_WORDS
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.tversky import tversky_similarity
from utils.post_processing import eurm_to_recommendation_list
from utils.pre_processing import bm25_row
import gc


dr = Datareader(mode='offline', verbose=False, only_load=True)
evaluator = Evaluator(dr)

# best: norm, wor, split, skipw, porter2, lanca2

norm=True
work=True
split=True
skip_words=True
date=False
porter=False
porter2=True
lanca=False
lanca2=True
data1 = False


nome = ""
if norm: nome+="norm_"
if work: nome+="work_"
if split: nome+="split_"
if skip_words: nome+="skipw_"
if porter: nome+="porter_"
if porter2: nome+="porter2_"
if lanca: nome+="lanca_"
if lanca2: nome+="lanca2_"
if data1: nome+="data1_"

nlp = NLP2(dr, stopwords=[], norm=norm,work=work,split=split,date=date, skip_words=skip_words,
           porter=porter,porter2=porter2,lanca=lanca,lanca2=lanca2)
# new_titles, occ_full, occ_single = nlp.fit( verbose=False, workout=True, normalize=True, date=True, lancaster=False,
#                                                     porter=False, underscore=True, double_fit=False)


ucm = nlp.get_UCM(data1=data1)
urm = dr.get_urm()
test_playlists = dr.get_test_pids()
print('ucm', ucm.shape)
print('Computing similarity...')
start = time.time()
# Compute similarity
ucm= bm25_row(ucm)

similarity = tversky_similarity(ucm, binary=False, shrink=1, alpha=0.1, beta=1)
similarity = similarity.tocsr()
print(time.time() - start)


print('Computing eurm...')
start = time.time()
# Compute eurm
eurm = dot_product(similarity, urm, k=500)
eurm = eurm.tocsr()
eurm = eurm[test_playlists, :]
print('eurm', eurm.shape)
print(time.time() - start)


# Evaluating
rec_list = eurm_to_recommendation_list(eurm)

sps.save_npz("nlp_eurm_online_bm25.npz", eurm, compressed=False)
np.save("nlp_rec_list_online_bm25",rec_list)

evaluator.evaluate(rec_list, name='AAANLP_bm25_'+nome, verbose=True, show_plot=False)





