import fim
from fim import fpgrowth
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from fim import apriori, fpgrowth, fim, arules
from utils.post_processing import rec_list_to_eurm, eurm_to_recommendation_list, eurm_remove_seed
from utils.evaluator import Evaluator
from utils.datareader import Datareader
from utils.definitions import *
from multiprocessing import Process, Pool


def build_test_dict(dr):
    print("building test dict", end=" ")
    test_interactions_df = dr.get_df_test_interactions()
    test_interactions_df.sort_values(['pos'], ascending=True)
    test_playlists_df = dr.get_df_test_playlists()
    test_playlists = test_playlists_df['pid'].values

    # A list of list [pos, tid] for each playlist sorted by pos
    test_known_tracks = test_interactions_df.groupby(['pid'])[['pos', 'tid']].apply(lambda x: x.values.tolist())
    for s in test_known_tracks:
        s = s.sort(key=lambda x: x[0])
    print("> done")
    return test_known_tracks


def urm_to_sequences_from_one_target(urm_pos, urm_pos_csc, song_target, list_of_list_of_lists=False):
    not_empty_lines = urm_pos_csc[:, song_target].nonzero()[0]
    filtered = urm_pos[not_empty_lines]
    sequences_spm = []
    for row in range(filtered.shape[0]):
        to_append = list(filtered.indices[filtered.indptr[row]:filtered.indptr[row + 1]]
                         [np.argsort(filtered.data[filtered.indptr[row]:filtered.indptr[row + 1]])])
        if list_of_list_of_lists:
            sequences_spm.append([[i] for i in to_append])
        else:
            sequences_spm.append(to_append)
    return sequences_spm


def fast_argpart(arr):
    if len(arr) > 500:
        max_n = 500
    else:
        max_n = len(arr)
    return np.argpartition(arr, -max_n)[-max_n:]


dr = Datareader(mode='offline', verbose=False, only_load=True)
ev = Evaluator(dr)

test_known_tracks = build_test_dict(dr)
test_pids_cat2 = dr.get_test_pids(cat=2)

urm_pos = dr.get_position_matrix(position_type='last')
urm_pos_csc = sps.csc_matrix(urm_pos)

###### NON FARE QUESTA CELLA
# for i in tqdm(range(1000, 2000)):
#     song_target = test_known_tracks[test_pids_cat2[i - 1000]][0][1]
#     not_empty_lines = urm_pos_csc[:, song_target].nonzero()[0]
#     filtered = urm_pos[not_empty_lines]
#     sequences_spm = []
#     for row in range(filtered.shape[0]):
#         to_append = list(filtered.indices[filtered.indptr[row]:filtered.indptr[row + 1]]
#                          [np.argsort(filtered.data[filtered.indptr[row]:filtered.indptr[row + 1]])])
#         sequences_spm.append(to_append)
#     save_obj(name="sequences_cat1_" + str(i), obj=sequences_spm, path=ROOT_DIR + '/data/cat1/')

costante_di_popolarita = 15

pred_lil = sps.lil_matrix((10000, 2262292))

for i in tqdm(range(1000,2000)):
    sequences = load_obj(path=ROOT_DIR+'/data/cat1/', name='sequences_cat1_'+str(i))
    popularity = len(sequences)
    preds_line = np.zeros(2262292)

    for seq in fpgrowth(sequences,supp= -popularity/costante_di_popolarita, target='m'):
        for song in seq[0]:
            preds_line[song]+= seq[1]*(len(seq[0])-1)*(len(seq[0])-1)
    vals = fast_argpart(preds_line)

    pred_lil[i,vals] = preds_line[vals]


eurm = sps.csr_matrix(pred_lil)
eurm = eurm_remove_seed(eurm , dr )
rec_list = eurm_to_recommendation_list(eurm)
ev.evaluate(rec_list, "cat2_spm_max",verbose=True, do_plot=True, show_plot=True, save=True )

exit()

# # parallel association rule.


import gc

target = 'm'
costante_di_pop = 15


# In[9]:


def association_rule(i):
    sequences = load_obj(path=ROOT_DIR + '/data/cat1/', name='sequences_cat1_' + str(i))
    popularity_iniziale = len(sequences)
    preds_line = np.zeros(2262292)

    if popularity_iniziale > 2000:
        mean_len = 0
        for seq in sequences:
            mean_len += len(seq)
        mean_len = mean_len / len(sequences)

        count = 0
        for j in range(len(sequences)):
            if len(sequences[j]) > (mean_len * 2) or len(sequences[j]) < (mean_len / 2):
                sequences[j] = []
                count += 1
        popularity = popularity_iniziale - count

        print(i, "iniziale",popularity_iniziale, "new_pop", popularity, "rimosse", count, " mean_l", mean_len, "num_seq", len(sequences))

        if popularity > 2000:
            mean_len = 0
            for seq in sequences:
                mean_len += len(seq)
            mean_len = mean_len / len(sequences)

            count = 0
            for j in range(len(sequences)):
                if len(sequences[j]) > (mean_len * 2) or len(sequences[j]) < (mean_len / 2):
                    sequences[j] = []
                    count += 1
            popularity -= count

            print(i, popularity_iniziale, "new_pop", popularity, "rimosse", count, " mean_l", mean_len, "num_seq",
                  len(sequences))

        if popularity > 2000:
            mean_len = 0
            for seq in sequences:
                mean_len += len(seq)
            mean_len = mean_len / len(sequences)

            count = 0
            for j in range(len(sequences)):
                if len(sequences[j]) > (mean_len * 2) or len(sequences[j]) < (mean_len / 2):
                    sequences[j] = []
                    count += 1
            popularity -= count
            print(i, popularity_iniziale, "new_pop", popularity, "rimosse", count, " mean_l", mean_len, "num_seq",
                  len(sequences))

    sequences = np.array(sequences)
    sequences = sequences[len(sequences) > 0]
    const = costante_di_pop

    sequences = fpgrowth(sequences, supp=-popularity / const, target=target)

    for seq in sequences:
        for song in seq[0]:
            preds_line[song] += seq[1] * (len(seq[0]) - 1) * (len(seq[0]) - 1)
    indices = fast_argpart(preds_line)

    preds_line_lil = sps.lil_matrix((1, 2262292))
    vals = fast_argpart(preds_line)
    preds_line_lil[0, vals] = preds_line[vals]

    del sequences, indices, preds_line, vals,
    gc.collect()
    print("nnz", preds_line_lil.nnz)

    return preds_line_lil

p = Pool(2)
pred_lil = sps.lil_matrix((10000, 2262292))
roba = p.map(association_rule, [x + 1000 for x in range(1000)])


pred_lil = sps.lil_matrix((10000, 2262292))
for prog, preds in enumerate(roba):
    pred_lil[prog + 1000] = preds



eurm = sps.csr_matrix(pred_lil)
eurm = eurm_remove_seed(eurm, dr)
rec_list = eurm_to_recommendation_list(eurm)
ev.evaluate(rec_list, "cat2_spm_max", verbose=True, do_plot=True, show_plot=True, save=True)





