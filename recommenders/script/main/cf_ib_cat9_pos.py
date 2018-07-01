from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True
filename = mode+'_npz/'+'cf_ib_'+mode+'.npz'

if len(arg)>1: eurm_k= int(arg[1])
else: eurm_k = 750

configs =[
    {'cat':2, 'alpha':1.0, 'beta':0, 'k':500, 'shrink':0,'threshold':0 },
    {'cat':3, 'alpha':0.7, 'beta':0, 'k':1800, 'shrink':0,'threshold':0 },
    {'cat':4, 'alpha':0.5, 'beta':0, 'k':1700, 'shrink':0,'threshold':0 },
    {'cat':5, 'alpha':0.6, 'beta':0, 'k':1200, 'shrink':0,'threshold':0 },
    {'cat':6, 'alpha':0.6, 'beta':0, 'k':1100, 'shrink':0,'threshold':0 },
    {'cat':7, 'alpha':0.6, 'beta':0, 'k':400, 'shrink':0,'threshold':0 },
    ]

configs_pos = [
    {'cat':9, 'alpha':0.8, 'beta':0, 'k':1100, 'shrink':0,'threshold':0 },
    ]

configs_strange= [
    {'cat':8, 'alpha':1, 'beta':0, 'k':1650, 'shrink':0,'threshold':0 }, #alpha1.1 ndcg rp no click
    {'cat':10,'alpha':1, 'beta':0, 'k':200, 'shrink':0,'threshold':0 },
    ]

#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)

urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
eurm = sp.csr_matrix(urm.shape)
rec = CF_IB_BM25(urm=urm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)
for c in configs:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], shrink=c['shrink'], threshold=c['threshold'])
    rec.recommend(target_pids=pids, eurm_k=eurm_k)
    rec.clear_similarity()
    eurm = eurm + rec.eurm
    rec.clear_eurm()

pos_m = dr.get_position_matrix(position_type='last')
pos_m = pre.norm_max_row(pos_m.tocsr())
for c in configs_pos:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], shrink=c['shrink'], threshold=c['threshold'])
    rec.urm = pos_m
    rec.recommend(target_pids=pids, eurm_k=eurm_k)
    rec.clear_similarity()
    eurm = eurm + rec.eurm
    rec.clear_eurm()

urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
rec = CF_IB_BM25_strange(urm=urm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)
for c in configs_strange:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], shrink=c['shrink'], threshold=c['threshold'])
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