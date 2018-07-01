from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True
filename = mode+'_npz/'+'cr_cb_al_ar_'+mode+'.npz'


eurm_k=750

### for ar al
configs =[
    {'cat':2, 'alpha':1, 'beta':0, 'k':300, 'shrink':0,'threshold':0 },
    {'cat':3, 'alpha':1, 'beta':0, 'k':155, 'shrink':0,'threshold':0 },
    {'cat':4, 'alpha':1, 'beta':0, 'k':155, 'shrink':0,'threshold':0 },
    {'cat':5, 'alpha':1, 'beta':0, 'k':155, 'shrink':0,'threshold':0 },
    {'cat':6, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':7, 'alpha':1, 'beta':0, 'k':100, 'shrink':0,'threshold':0 },
    {'cat':8, 'alpha':1, 'beta':0, 'k':75, 'shrink':0,'threshold':0 },
    {'cat':9, 'alpha':1, 'beta':0, 'k':50, 'shrink':0,'threshold':0 },
    {'cat':10,'alpha':1, 'beta':0, 'k':50, 'shrink':0,'threshold':0 },
    ]

#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)
urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)

feats = ['acousticness_cluster','danceability_cluster','duration_ms_cluster','energy_cluster','instrumentalness_cluster',
             'liveness_cluster','loudness_cluster','speechiness_cluster','tempo_cluster','valence_cluster','popularity_cluster']
icm = sp.csr_matrix((2262292, 0))
icm_al = dr.get_icm(arid=False, alid=True)
icm = sp.hstack([icm, icm_al])
for feat in feats:
        icm_feat = dr.get_icm_refined_feat(feat=feat, K=4, load_only=True)
        icm = sp.hstack([icm, icm_feat, icm_al])
    
rec = CB_AL_AR_BM25(urm=urm, icm=icm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)

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
