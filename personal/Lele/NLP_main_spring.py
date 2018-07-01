#python NLP_main.py True True True True True True True True True True
import sys
# assert len(sys.argv) == 11, "devi dare 10 parametri "

import time
from utils.datareader import Datareader
from utils.evaluator import Evaluator
import pandas as pd
from personal.Lele.NLP import NLP2
from utils.definitions import ROOT_DIR
from utils.definitions import STOP_WORDS
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.tversky import tversky_similarity
from utils.post_processing import eurm_to_recommendation_list
import gc
import numpy as np


norm=bool(sys.argv[1])
work=bool(sys.argv[2])
split=bool(sys.argv[3])
skip_words=bool(sys.argv[4])
date=bool(sys.argv[5])
porter=bool(sys.argv[6])
porter2=bool(sys.argv[7])
lanca=bool(sys.argv[8])
lanca2=bool(sys.argv[9])
data1 = bool(sys.argv[10])

dr = Datareader(mode='offline', verbose=False, only_load=True)
evaluator = Evaluator(dr)


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
(prec_t, ndcg_t, clicks_t, prec_a, ndcg_a, clicks_a) = evaluator.evaluate(rec_list, return_overall_mean=True ,name='AAANLP_'+nome, verbose=True, show_plot=False)

# gc.collect()
# del eurm, rec_list, similarity, nlp, test_playlists, start, nome
# gc.collect()

np.save("ret",[clicks_t])
