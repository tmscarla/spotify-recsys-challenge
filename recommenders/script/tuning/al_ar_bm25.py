from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline','filename','x','1,2,0.5']

mode = arg[0]

if mode=='tuning':
    dr_mode='offline'
    mode_tuning = arg[1]
    start, end, step = [float(x) for x in arg[2].split(',')]
    filename = arg[3]
elif mode=='offline': dr_mode='offline'
elif mode=='online': dr_mode='online'

weight_ar=0.65
weight_al=1.1

dr = Datareader(mode=dr_mode, only_load=True, verbose=False)
urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
icm_ar = dr.get_icm(arid=True,alid=False)
icm_ar.data = np.full(icm_ar.data.shape[0],weight_ar)
icm_al = dr.get_icm(arid=False,alid=True)
icm_al.data = np.full(icm_al.data.shape[0],weight_al)
icm = sp.hstack((icm_ar.tocsr(),icm_al.tocsr()))
pids = dr.get_test_pids()

rec = CB_AL_AR_BM25(icm=icm,urm=urm, binary=True, datareader=dr, mode=dr_mode, verbose=True, verbose_evaluation= False)

if mode=='tuning':
    if mode_tuning=='k':
        rec.tune_k(alpha=1,shrink=0, range_k=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)

    elif mode_tuning=='shrink':
        rec.tune_shrink(k=160,alpha=0.75, range_shrink=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)
        
    elif mode_tuning=='alpha':
        rec.tune_alpha(k=160, shrink=0, range_alpha=np.arange(start,end,step), filename=filename,verbose_tune=True, overwrite=True)

if mode=='offline':
    rec.model(alpha=0.9, k=160, shrink=0, threshold=0)
    rec.fast_recommend()
    #rec.save_small_eurm('eurm_ar_bm25')
    mean, full = rec.fast_evaluate_eurm()
    print(full)

if mode=='online':
    #write here the best configuration found
    print('online not already implemented')