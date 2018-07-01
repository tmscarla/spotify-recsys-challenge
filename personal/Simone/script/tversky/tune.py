from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import  eurm_to_recommendation_list, eurm_remove_seed
import recommenders.similarity.s_plus as ss
import numpy as np
from utils import ensembler
import scipy.sparse as sps
import gc
from sklearn.preprocessing import normalize
import sys

dr = Datareader(verbose=False, mode='offline', only_load=True)
ev = Evaluator(dr)

#Getting for the recommender algorithm
urm = dr.get_urm()
pids = dr.get_test_pids()
t_urm = sps.csr_matrix(urm[pids])

def recsys(shrink):
        alpha = 0.25
        beta = 0.65
        k = 200
        config = ('alpha=%.2f beta=%.2f k=%d shrink=%d binary=False' % (alpha, beta, k ,shrink))
        #print(config)
        sim = ss.tversky_similarity(urm.T, urm, k=k, alpha=alpha, beta=beta, shrink=shrink, binary=False, verbose=True)
        #Computing ratings and remove seed
        eurm = ss.dot_product(t_urm, sim.T, k=750)
        del sim
        eurm = eurm_remove_seed(eurm, dr)
        #evaluation
        res = ev.evaluate(eurm_to_recommendation_list(eurm), 'ciao', verbose=False)        
        del eurm
        return res[0:3], config

results = []

with open('result_tversky_shrink2','w') as file:
    for shrink in np.arange(120,200,20):
            res, config = recsys(shrink)
            file.write(str(config)+'\n'+str(res)+'\n')
            print(config)
            print(res)
            