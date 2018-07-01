from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True

filename = mode+'_npz/'+'cr_cf_al_'+mode+'.npz'

eurm_k=750
configs_strange =[
    {'cat':2, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':3, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':4, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':5, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':6, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':7, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':8, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':9, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':10,'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    ]

#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)

urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
eurm = sp.csr_matrix(urm.shape)
icm = sp.csr_matrix((2262292, 0))  # empty dummy, and stacking features on it
feats = ['alid']
for feat in feats:
    icm_feat = dr.get_icm_refined_pid_feat(feat=feat, K=4, load_only=True,mode=mode)
    icm = sp.hstack([icm, icm_feat])
rec = CF_IB_BM25(urm=icm.T, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)
for c in configs_strange:
    pids = dr.get_test_pids(cat=c['cat'])
    rec.model(alpha=c['alpha'],beta=c['beta'], k=c['k'], shrink=c['shrink'], threshold=c['threshold'])
    rec.urm = urm
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
