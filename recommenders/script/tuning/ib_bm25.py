from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['tuning','alpha','0.8,1.3,0.1','prova2']

mode = arg[0]

if mode=='tuning':
    dr_mode='offline'
    mode_tuning = arg[1]
    start, end, step = [float(x) for x in arg[2].split(',')]
    filename = arg[3]
elif mode=='offline': dr_mode='offline'
elif mode=='online': dr_mode='online'


dr = Datareader(mode=dr_mode, only_load=True, verbose=False)
urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)

rec = CF_IB_BM25(urm=urm, binary=True, datareader=dr, mode=dr_mode, verbose=True, verbose_evaluation= False, mode_t=True)

if mode=='tuning':
    if mode_tuning=='k':
        rec.tune_k(alpha=1,shrink=0, beta=0, range_k=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)

    elif mode_tuning=='shrink':
        rec.tune_shrink(k=250,alpha=1, beta=0, range_shrink=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)
        
    elif mode_tuning=='alpha':
        rec.tune_alpha(k=1000, shrink=0, range_alpha=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)
    
    elif mode_tuning=='beta':
        rec.tune_beta(k=1000, shrink=0, alpha=1, range_beta=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)
    
    elif mode_tuning=='alphabeta':
        rec.tune_alpha_beta(k=200, shrink=0, range_alpha=np.arange(0.1,2.1,0.1), range_beta=np.arange(0.1,2.1,0.1), filename=filename,verbose_tune=True, overwrite=True)

if mode=='offline':
    rec.model(alpha=1, k=250, shrink=0, threshold=0)
    rec.fast_recommend()
    #rec.save_small_eurm('eurm_ar_bm25')
    mean, full = rec.fast_evaluate_eurm()
    print(full)

if mode=='online':
    #write here the best configuration found
    print('online not already implemented')