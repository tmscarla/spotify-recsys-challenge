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
        config = ('alpha=0.4 k=200 shrink=%d binary=False' % (shrink))
        print(config)
        sim = ss.cosine_similarity(urm.T, urm, k=200, alpha=0.4  ,shrink=shrink, binary=False, verbose=True)
        #Computing ratings and remove seed
        eurm = ss.dot_product(t_urm, sim.T, k=750)
        del sim
        eurm = eurm_remove_seed(eurm, dr)
        #evaluation
        res = ev.evaluate(eurm_to_recommendation_list(eurm), 'ciao', verbose=False)        
        del eurm
        return res[0:3], config

results = []

with open('result_cosine_shrink','w') as file:
    for shrink in np.arange(0,40,5):
        res, config = recsys(shrink)
        file.write(str(config)+'\n'+str(res)+'\n')
        print(config)
        print(res)
    for shrink in np.arange(40,300,20):
        res, config = recsys(shrink)
        file.write(str(config)+'\n'+str(res)+'\n')
        print(config)
        print(res)
