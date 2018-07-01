from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.post_processing import  eurm_to_recommendation_list_submission
from utils.post_processing import  eurm_to_recommendation_list, eurm_remove_seed
import recommenders.similarity.s_plus as ss
import recommenders.similarity.p3alpha_rp3beta as p3r3
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
#urm.data = np.ones(len(urm.data))
p_ui = normalize(urm, norm="l1")
p_iu = normalize(urm.T, norm="l1")
pop = urm.sum(axis=0).A1
pids = dr.get_test_pids()
t_urm = sps.csr_matrix(p_ui.copy()[pids])

def recsys(alpha, beta):
        alpha = alpha
        beta = beta
        k = 200
        shrink = 100
        config = ('alpha=%.2f beta=%.2f k=%d shrink=%d binary=False' % (alpha, beta, k ,shrink))
        #print(config)
        sim = p3r3.p3alpha_rp3beta_similarity(p_iu, p_ui, pop, k= k, shrink=shrink, alpha=alpha, beta=beta, verbose=True, mode=1)
        #Computing ratings and remove seed
        eurm = ss.dot_product(t_urm, sim, k=750)
        del sim
        eurm = eurm_remove_seed(eurm, dr)
        #evaluation
        res = ev.evaluate(eurm_to_recommendation_list(eurm), 'ciao', verbose=False)        
        del eurm
        return res[0:3], config

results = []

res, config = recsys(0.5,0.4)
print(config)
print(res)
'''
with open('result_p3r3','w') as file:
    for alpha in np.arange(0,1.1,0.1):
            for beta in np.arange(0,1.1,0.1):
                res, config = recsys(alpha,beta)
                file.write(str(config)+'\n'+str(res)+'\n')
                print(config)
                print(res)
'''