from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True
filename = mode+'_npz/'+'cb_al_'+mode+'.npz'

if len(arg)>1: eurm_k= int(arg[1])
else: eurm_k = 750

configs =[
    {'cat':2, 'alpha':1, 'beta':0, 'k':60, 'shrink':0,'threshold':0 },
    {'cat':3, 'alpha':0.75, 'beta':0, 'k':50, 'shrink':0,'threshold':0 },
    {'cat':4, 'alpha':0.7, 'beta':0, 'k':80, 'shrink':0,'threshold':0 },
    {'cat':5, 'alpha':1, 'beta':0, 'k':70, 'shrink':0,'threshold':0 },
    {'cat':6, 'alpha':1, 'beta':0, 'k':60, 'shrink':0,'threshold':0 },
    {'cat':7, 'alpha':1, 'beta':0, 'k':50, 'shrink':0,'threshold':0 },
    {'cat':8, 'alpha':0.7, 'beta':0, 'k':65, 'shrink':0,'threshold':0 },
    {'cat':9, 'alpha':0.8, 'beta':0, 'k':65, 'shrink':0,'threshold':0 },
    {'cat':10,'alpha':0.8, 'beta':0, 'k':50, 'shrink':0,'threshold':0 },
    ]


#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)
urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
icm = dr.get_icm(arid=False,alid=True)
rec = CB_AL_BM25(urm=urm, icm=icm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)

eurm = sp.csr_matrix(urm.shape)

for c in configs:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'], k=c['k'], shrink=c['shrink'], threshold=c['threshold'])
    rec.recommend(target_pids=pids, eurm_k=eurm_k)
    rec.clear_similarity()
    eurm = eurm + rec.eurm
    rec.clear_eurm()

pids = dr.get_test_pids()
eurm = eurm[pids]

if mode=='offline':
    rec_list = post.eurm_to_recommendation_list(eurm=eurm, datareader=dr, remove_seed=True, verbose=False)
    mean, full = rec.ev.evaluate(rec_list, str(rec) , verbose=True, return_result='all')

if save:
    sp.save_npz(filename ,eurm)