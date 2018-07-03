from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True
filename = mode+'_npz/'+'cf_al_'+mode+'.npz'

if len(arg)>1: eurm_k= int(arg[1])
else: eurm_k = 750

configs_mix =[
    {'cat':2, 'alpha':1, 'beta':0.1, 'power':1.1, 'k':1500, 'shrink':0,'threshold':0 },
    {'cat':3, 'alpha':1, 'beta':0.1, 'power':2.8, 'k':300, 'shrink':0,'threshold':0 },
    {'cat':4, 'alpha':1, 'beta':0.1, 'power':1.4, 'k':300, 'shrink':0,'threshold':0 },
    {'cat':5, 'alpha':1, 'beta':0.1, 'power':2.2, 'k':200, 'shrink':0,'threshold':0 },
    {'cat':6, 'alpha':1, 'beta':0.1, 'power':2.6, 'k':125, 'shrink':0,'threshold':0 },
    {'cat':8, 'alpha':1, 'beta':0.1, 'power':3.0, 'k':125, 'shrink':0,'threshold':0 },
    {'cat':10,'alpha':1, 'beta':0.1, 'power':2.7, 'k':250, 'shrink':0,'threshold':0 },
    ]

configs_pos_matrix =[
    {'cat':7, 'alpha':1, 'beta':0.1, 'power':1.5, 'k':150, 'shrink':0,'threshold':0 },
    {'cat':9, 'alpha':1, 'beta':0.1, 'power':1.4, 'k':200, 'shrink':0,'threshold':0 },
    ]



#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)

dr = Datareader(mode=mode, only_load=True, verbose=False)
urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
urm.data = np.ones(urm.data.shape[0])
icm = sp.csr_matrix(dr.get_icm(alid=True, arid=False),dtype=np.float)
ucm = urm*icm

rec = CF_AL_BM25(urm=urm, ucm=ucm, binary=False, bm25=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False, similarity='tversky')

eurm = sp.csr_matrix(urm.shape)

for c in configs_mix:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], power=c['power'],
    shrink=c['shrink'], threshold=c['threshold'],target_items=pids)
    rec.recommend(target_pids=pids, eurm_k=eurm_k)
    rec.clear_similarity()
    eurm = eurm + rec.eurm
    rec.clear_eurm()

urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
pos_m = dr.get_position_matrix(position_type='last')
pos_m = pre.norm_max_row(pos_m.tocsr())
rec = CF_UB_BM25(urm=urm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False, similarity='tversky')
for c in configs_pos_matrix:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], power=c['power'],
    shrink=c['shrink'], threshold=c['threshold'],target_items=pids)
    #inject the position matrix
    rec.urm = pos_m
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