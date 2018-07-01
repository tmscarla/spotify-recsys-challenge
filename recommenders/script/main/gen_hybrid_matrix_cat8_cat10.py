from utils.audio_feature_cluster import *
import pandas as pd
import numpy  as np
from tqdm import tqdm
import scipy.sparse as sp
from utils.definitions import ROOT_DIR
from utils.datareader import Datareader

"""
This file is used to generate the hybrid icm for cat8 and cat10.
"""

import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]


icm_cat8 = sp.csr_matrix((2262292, 0))
icm_cat10 = sp.csr_matrix((2262292, 0))

######################################################################################################
## Generate layered pid with artist and album
###################################################################################################### 

dr = Datareader(mode=mode, only_load=True, verbose=False)
train_intr = dr.get_df_train_interactions()
feats = ['alid', 'arid']
for feat in tqdm(feats, desc='Generating layered playlist'):
    af = pd.read_csv(ROOT_DIR + '/data/original/tracks.csv', sep='\t',usecols=['tid', feat], dtype={feat: str})
    # merge
    df = train_intr.merge(af, left_on='tid', right_on='tid', how='inner')
    del af
    # refine
    df['pid_'+feat] = df['pid'].astype(str).str.cat(df[feat].astype(str), sep='-')
    # prepare unique ids for 'pid_feat'
    df_tmp = pd.DataFrame()
    df_tmp['pid_'+feat] = df['pid_'+feat]
    df_tmp = df_tmp.drop_duplicates()
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['new_pid'] = df_tmp.index
    # attact new_pid to tid
    df = df.merge(df_tmp, left_on='pid_'+feat, right_on='pid_'+feat, how='inner')
    del df_tmp
    # start building icm
    n_tracks = 2262292
    trs = df['tid'].values
    n = len(df)
    pids = df['new_pid'].values
    del df
    n_pids = pids.max() + 1  # index starts from 0
    # create partial icm 
    icm_ = sp.csr_matrix((np.ones(n), (trs, pids)), shape=(n_tracks, n_pids),
                           dtype=np.int32)
    
    icm_cat8 = sp.hstack([icm_cat8, icm_])
    icm_cat10 = sp.hstack([icm_cat10, icm_])


urm = dr.get_urm()
icm_pl = urm.copy().T
icm_al = dr.get_icm(arid=False, alid=True)
icm_ar = dr.get_icm(arid=True, alid=False)

icm_cat8 = sp.hstack([icm_cat8, icm_al, icm_ar])
icm_cat10 = sp.hstack([icm_cat10, icm_al, icm_ar])

# hybrid cat8
for i in range(0,5):
    icm_cat8 = sp.hstack([icm_cat8, icm_pl])
# hybrid cat10
for i in range(0,3):
    icm_cat10 = sp.hstack([icm_cat10, icm_pl])

# dump the icms

sp.save_npz(ROOT_DIR + '/data/hybrid_icm_cat8_'+mode+'.npz',icm_cat8)
sp.save_npz(ROOT_DIR + '/data/hybrid_icm_cat10_'+mode+'.npz',icm_cat10)
