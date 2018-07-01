from utils.datareader import Datareader
from utils.evaluator import  Evaluator
from utils.submitter import Submitter
from utils.print_tuning import TunePrint
import utils.post_processing as post
import utils.pre_processing as pre
import recommenders.similarity.s_plus as ss
import recommenders.similarity.p3alpha_rp3beta as p3r3
import numpy as np
import scipy.sparse as sps
from recommenders.nlp import NLP

       
#similarity = tversky_similarity(ucm, binary=False, shrink=1, alpha=0.1, beta=1


class NLP_BM25:
        def __init__(self, urm, ucm=None, stopwords=[], load_ucm=False, save_ucm=False, binary=False, verbose=True, mode='offline', datareader=None, verbose_evaluation=True):
                assert(mode in ('offline', 'online'))
                if binary: urm.data=np.ones(urm.data.shape[0])
                # best: norm, wor, split, skipw, porter2, lanca2
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
                self.ucm=ucm
                if self.ucm is None and not load_ucm:
                        nlp = NLP(datareader, stopwords=stopwords, norm=norm, work=work, split=split, date=date, skip_words=skip_words,
                        porter=porter, porter2=porter2, lanca=lanca, lanca2=lanca2)
                        self.ucm = nlp.get_UCM(data1=data1)
                elif self.ucm is None and load_ucm:
                        self.load_ucm('ucm_nlp.npz')
                if save_ucm:
                        self.save_ucm('ucm_nlp.npz')
                self.m_uc = pre.bm25_row(self.ucm.copy()).tocsr()
                self.m_cu = pre.bm25_row(self.ucm.copy()).T.tocsr()
                self.urm = urm
                self.binary = binary
                self.verbose = verbose
                self.verbose_ev = verbose_evaluation
                self.dr = datareader
                self.mode = mode
                if mode == 'offline':
                        self.ev = Evaluator(self.dr)
        
        def model(self, alpha=1, k=200, shrink=0, threshold=0, target_items=None):
                if target_items is None: target_items=self.dr.get_test_pids() # work with s*urm
                self.alpha = alpha
                self.k = k
                self.shrink, self.threshold = shrink, threshold
                self.s = ss.p3alpha_similarity(self.m_uc, self.m_cu,
                        k=k, shrink=shrink, alpha=alpha, threshold=threshold,
                        verbose=self.verbose, target_items=target_items)
        
        def recommend(self, target_pids=None, eurm_k=750):
                #if target_pids is None it calculate the whole eurm
                self.eurm = ss.dot_product(self.s, self.urm, k=eurm_k, target_items=target_pids, verbose=self.verbose)
                # TODO: here we can try some postprocessing on eurm if complete (like normalize for column)
        
        #### METHODS FOR OFFLINE MODE ####
        def fast_recommend(self, target_pids=None, eurm_k=750):
                assert(self.mode=='offline')
                if target_pids is None: target_pids=self.dr.get_test_pids()
                self.recommend(target_pids=target_pids, eurm_k=eurm_k)

        def fast_evaluate_eurm(self, target_pids=None):
                assert(self.mode=='offline')
                res = self.ev.fast_evaluate_eurm(self.eurm, target_pids=target_pids, verbose=self.verbose_ev)
                return res

        def evaluate_eurm(self, target_pids):
                assert(self.mode=='offline')
                eurm = sps.csr_matrix(self.eurm[target_pids])
                eurm = post.eurm_remove_seed(eurm, self.dr)
                rec_list = post.eurm_to_recommendation_list(eurm)
                res = self.ev.evaluate(rec_list, str(self) , verbose=self.verbose_ev, return_result='all')
                return res

        #### UTILITY METHODS ####
        
        def clear_similarity(self): del self.s

        def clear_eurm(self): del self.eurm

        def save_similarity(self, name_file, compressed=False):
                sps.save_npz(name_file, self.s, compressed)

        def save_small_eurm(self, name_file, target_pids, compressed=True):
                eurm = sps.csr_matrix(self.eurm[target_pids])
                sps.save_npz(name_file, eurm, compressed)
                
        def save_ucm(self, name_file, compressed=False):
                sps.save_npz(name_file, self.ucm.tocsr(), compressed)
        
        def load_ucm(self, name_file):
                self.ucm = sps.load_npz(name_file).tocsr()

        #### OVERRIDE METHODS ####

        def __str__(self):
                name = ('NLP_BM25: alpha=%.3f, beta=%.3f, k=%d, shrink=%d, threshold=%.5f, binary=%s, rp3mode=%d' 
                        % (self.alpha, self.beta, self.k , self.shrink, self.threshold, str(self.binary), self.rp3_mode))
                return name
        
        #### TUNING METHODS ####

