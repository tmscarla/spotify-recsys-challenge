import pandas as pd
import numpy as np
import nltk
from nltk import stem
from nltk.tokenize import RegexpTokenizer
from utils.datareader import Datareader
from tqdm import tqdm
from scipy import sparse
from difflib import SequenceMatcher
from difflib import get_close_matches
from utils.pre_processing import *
from recommenders.similarity.dot_product import dot_product
from recommenders.similarity.s_plus import tversky_similarity
from utils.evaluator import Evaluator
from utils.post_processing import *
from personal.Tommaso.NLP.NLP import NLP
from utils.definitions import *

# Datareader
dr = Datareader(mode='online', only_load=True)
#ev = Evaluator(dr)

# Dataframe with interactions
df_train = dr.get_df_train_interactions()
df_test = dr.get_df_test_interactions()
df = pd.concat([df_train, df_test], axis=0, join='outer')

playlists = df['pid'].as_matrix()
tracks = df['tid'].as_matrix()
dictionary = dr.get_track_to_artist_dict()

pids = list(dr.get_train_pids()) + list(dr.get_test_pids())

# URM
urm = dr.get_urm()
urm = urm[pids]
print(urm.shape)

print('artists...')
artists = [dictionary[t] for t in tracks]

print('ucm...')
ucm = sparse.csr_matrix((np.ones(len(playlists)), (playlists, artists)), shape=(1049361, len(dr.get_artists())))
ucm = ucm.tocsr()
ucm = ucm[pids]
print(ucm.shape)


ucm = bm25_row(ucm)


print('similarity..')
sim = tversky_similarity(ucm, ucm.T, shrink=200, alpha=0.1, beta=1, k=800, verbose=1, binary=False)
sim = sim.tocsr()

test_pids = list(dr.get_test_pids())

eurm = dot_product(sim, urm, k=750)
eurm = eurm.tocsr()
eurm = eurm[test_pids, :]
sparse.save_npz('eurm_artists.npz', eurm)

#ev.evaluate(eurm_to_recommendation_list(eurm), name='cbf_user_artist', show_plot=False)

exit()



