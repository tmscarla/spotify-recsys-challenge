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
import utils.sparse as ut

class CF_IB_BM25_strange:
        def __init__(self, urm, pop=None, binary=False,K1=1.2, B=0.75, verbose=True, mode='offline', datareader=None, verbose_evaluation=True, mode_t=False, trick=False):
                assert(mode in ('offline', 'online'))
                if binary: urm.data=np.ones(urm.data.shape[0])
                if pop is None: self.pop = urm.sum(axis=0).A1
                else: self.pop = pop
                self.dr = datareader
                self.urm = urm
                urm_aux = urm.copy()
                ut.inplace_set_rows_zero(X=urm_aux,target_rows=self.dr.get_test_pids()) #don't learn from challange set
                urm_aux.eliminate_zeros()
                if mode_t: self.m_ui = urm_aux.copy().tocsr()
                else: self.m_ui = pre.bm25_row(urm_aux.copy(),K1=K1,B=B).tocsr()
                if mode_t: self.m_iu = urm_aux.T.copy().tocsr()
                else: self.m_iu = pre.bm25_row(urm_aux.T.copy(),K1=K1,B=B).tocsr()
                self.binary = binary
                self.verbose = verbose
                self.verbose_ev = verbose_evaluation
                self.mode = mode
                self.mode_t = mode_t
                if trick: self.urm = pre.bm25_row(urm).tocsr() #high click, high ndcg, better no use
                if mode == 'offline':
                        self.ev = Evaluator(self.dr)
        
        def model(self, alpha=1, beta=0, k=200, shrink=0, threshold=0, rp3_mode=1, target_items=None):
                #if target_items is None it calculate the whole similarity
                self.alpha, self.beta = alpha, beta
                self.k = k
                self.shrink, self.threshold = shrink, threshold
                self.rp3_mode = rp3_mode
                if self.mode_t:
                        self.s = ss.tversky_similarity(self.m_ui.T, self.m_iu.T,
                        k=k, shrink=shrink, alpha=alpha, beta=beta, threshold=threshold,
                        verbose=self.verbose, target_items=target_items)  
                elif beta==0:
                        self.s = ss.p3alpha_similarity(self.m_ui.T, self.m_iu.T,
                        k=k, shrink=shrink, alpha=alpha, threshold=threshold,
                        verbose=self.verbose, target_items=target_items)
                else:
                        self.s = p3r3.p3alpha_rp3beta_similarity(self.m_ui.T, self.m_iu.T, self.pop,
                        k=k, shrink=shrink, alpha=alpha, beta=beta, threshold=threshold,
                        verbose=self.verbose, mode=rp3_mode, target_items=target_items)
                
        
        def recommend(self, target_pids=None, eurm_k=750):
                #if target_pids is None it calculate the whole eurm
                self.eurm = ss.dot_product(self.urm, self.s.T, k=eurm_k, target_items=target_pids, verbose=self.verbose)
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
                
        #### OVERRIDE METHODS ####

        def __str__(self):
                name = ('CF_IB_BM25: alpha=%.3f, beta=%.3f, k=%d, shrink=%d, threshold=%.5f, binary=%s, rp3mode=%d' 
                        % (self.alpha, self.beta, self.k , self.shrink, self.threshold, str(self.binary), self.rp3_mode))
                return name
        
        #### TUNING METHODS ####

        def tune_alpha_beta(self, range_alpha=np.arange(0,1.1,0.1),range_beta=np.arange(0,1.1,0.1), 
                k=200, shrink=0, threshold=0, verbose_tune=True,
                filename='tuning_bm25_alpha_beta', overwrite=False, save_mean = True, save_full=True
                ):
                tp = TunePrint(filename=filename, full=save_full, mean=save_mean, overwrite=overwrite)
                for alpha in range_alpha:
                        for beta in range_beta:
                                self.model(alpha=alpha,beta=beta, k=k, shrink=shrink, threshold=threshold)
                                self.fast_recommend()
                                self.clear_similarity()
                                mean, df_all_values = self.fast_evaluate_eurm()
                                self.clear_eurm()
                                s_mean = 'P = %1.4f, NDCG = %1.4f, CLICK = %1.4f'%(mean[0],mean[1],mean[2])
                                if verbose_tune: print(str(self)+'\n'+s_mean)
                                # save values
                                if save_mean: tp.print_mean_values(str(self), mean)
                                if save_full: tp.print_full_values(description=str(self),dict_val={'alpha':alpha, 'beta':beta}, dataframe=df_all_values)
                tp.make_pdf_full()
        
        #use this tuning method only with beta=0
        def tune_alpha(self, range_alpha=np.arange(0.5,1.5,0.1), k=100,
                shrink=0, threshold=0, verbose_tune=False,
                filename='tuning_bm25_alpha', overwrite=False, save_mean = True, save_full=True
                ):
                tp = TunePrint(filename=filename, full=save_full, mean=save_mean, overwrite=overwrite)
                self.model(alpha=1, beta=0, k=k, shrink=shrink, threshold=threshold) #exploit this trick to generate fastest model
                save_data = self.s.data
                for alpha in range_alpha:
                        self.s.data = save_data
                        self.s.data = np.power(self.s.data, alpha)
                        self.alpha = alpha
                        self.fast_recommend()
                        mean, df_all_values = self.fast_evaluate_eurm()
                        self.clear_eurm()
                        s_mean = 'P = %1.4f, NDCG = %1.4f, CLICK = %1.4f'%(mean[0],mean[1],mean[2])
                        if verbose_tune: print(str(self)+'\n'+s_mean)
                        if save_mean: tp.print_mean_values(str(self), mean)
                        if save_full: tp.print_full_values(description=str(self),dict_val={'alpha':alpha}, dataframe=df_all_values)
                tp.make_pdf_full()
        
        def tune_beta(self, range_beta=np.arange(0.0,2,0.1), alpha=1, k=100,
                shrink=0, threshold=0, verbose_tune=True,
                filename='tuning_bm25_k', overwrite=False, save_mean = True, save_full=True
                ):
                tp = TunePrint(filename=filename, full=save_full, mean=save_mean, overwrite=overwrite)
                for beta in range_beta:
                        self.model(alpha=alpha, beta=beta, k=k, shrink=shrink, threshold=threshold)
                        self.fast_recommend()
                        self.clear_similarity()
                        mean, df_all_values = self.fast_evaluate_eurm()
                        self.clear_eurm()
                        s_mean = 'P = %1.4f, NDCG = %1.4f, CLICK = %1.4f'%(mean[0],mean[1],mean[2])
                        if verbose_tune: print(str(self)+'\n'+s_mean)
                        # save values
                        if save_mean: tp.print_mean_values(str(self), mean)
                        if save_full: tp.print_full_values(description=str(self),dict_val={'beta':k}, dataframe=df_all_values)
                tp.make_pdf_full()


        def tune_k(self, range_k=np.arange(25,300,25), alpha=1, beta=0,
                shrink=0, threshold=0, verbose_tune=True,
                filename='tuning_bm25_k', overwrite=False, save_mean = True, save_full=True
                ):
                tp = TunePrint(filename=filename, full=save_full, mean=save_mean, overwrite=overwrite)
                for k in range_k:
                        self.model(alpha=alpha, beta=beta, k=k, shrink=shrink, threshold=threshold)
                        self.fast_recommend()
                        self.clear_similarity()
                        mean, df_all_values = self.fast_evaluate_eurm()
                        self.clear_eurm()
                        s_mean = 'P = %1.4f, NDCG = %1.4f, CLICK = %1.4f'%(mean[0],mean[1],mean[2])
                        if verbose_tune: print(str(self)+'\n'+s_mean)
                        # save values
                        if save_mean: tp.print_mean_values(str(self), mean)
                        if save_full: tp.print_full_values(description=str(self),dict_val={'k':k}, dataframe=df_all_values)
                tp.make_pdf_full()               
        
        def tune_shrink(self, range_shrink=np.arange(25,300,25), alpha=1, beta=0, k=200,
                threshold=0, verbose_tune=True,
                filename='tuning_bm25_shrink', overwrite=False, save_mean = True, save_full=True
                ):
                tp = TunePrint(filename=filename, full=save_full, mean=save_mean, overwrite=overwrite)
                for shrink in range_shrink:
                        self.model(alpha=alpha, beta=beta, k=k, shrink=shrink, threshold=threshold)
                        self.fast_recommend()
                        self.clear_similarity()
                        mean, df_all_values = self.fast_evaluate_eurm()
                        self.clear_eurm()
                        s_mean = 'P = %1.4f, NDCG = %1.4f, CLICK = %1.4f'%(mean[0],mean[1],mean[2])
                        if verbose_tune: print(str(self)+'\n'+s_mean)
                        # save values
                        if save_mean: tp.print_mean_values(str(self), mean)
                        if save_full: tp.print_full_values(description=str(self),dict_val={'shrink':shrink}, dataframe=df_all_values)
                tp.make_pdf_full()

