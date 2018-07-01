import time
from tqdm import tqdm
import numpy as np
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from personal.Lele.NLP import NLP2

from utils.post_processing import rec_list_to_eurm,eurm_to_recommendation_list,eurm_remove_seed


def not_empty_lines_by_target(urm_pos, target_list, min_songs_in_common):
    tmp = urm_pos.tocsc(copy=True)
    tmp = tmp[:, target_list]
    tmp = tmp.tocsr()
    not_empty_lines = list()
    for i in tqdm(range(tmp.shape[0]), desc="eliminating cols"):
        if len(tmp.indices[tmp.indptr[i]:tmp.indptr[i + 1]]) >= min_songs_in_common :
            not_empty_lines.append(i)
    return not_empty_lines


def urm_to_sequences(urm_pos, target_list, min_common):
    sequences_spm = []

    not_empty_lines = not_empty_lines_by_target(urm_pos, target_list, min_common)
    filtered = urm_pos[not_empty_lines]
    for row in tqdm((range(filtered.shape[0])), desc='Converting eurm into list of lists'):
        to_append = list(filtered.indices[filtered.indptr[row]:filtered.indptr[row + 1]]
                             [np.argsort(filtered.data[filtered.indptr[row]:filtered.indptr[row + 1]])])
        sequences_spm.append( [[i] for i in to_append])
    return sequences_spm

if __name__ == '__main__':


    dr = Datareader(mode='offline',verbose=False, only_load=True)
    ev = Evaluator(datareader=dr)

    nlp = NLP2(dr, stopwords=[], norm=True, work=True, split=True, date=False, skip_words=True,
               porter=False, porter2=True, lanca=False, lanca2=True)


    ucm_csr = nlp.get_UCM(data1=True).tocsr()
    ucm_csc = ucm_csr.tocsc(copy=True)

    urm_csr = dr.get_urm().tocsr()
    urm_csc = urm_csr.tocsc(copy=True)

    test_playlists = dr.get_test_pids(cat=1)
    # test_playlists.extend(dr.get_test_pids(cat=2))

    rec_list = [[] for x in range(10000)]




    i=0
    for playlist_id in tqdm(test_playlists):
        tokens = ucm_csr.indices[ucm_csr.indptr[playlist_id]:ucm_csr.indptr[playlist_id+1]]
        playlists_with_tokens=[]
        for token in tokens:
            playlists_with_tokens.extend(ucm_csc.indices[ ucm_csc.indptr[token]:ucm_csc.indptr[token + 1]] )

        urm_tmp = urm_csr[playlists_with_tokens]

        track_total_interactions = np.array(urm_tmp.sum(axis=0)).astype(np.int32)[0, :]  # like ravel

        top_pop = track_total_interactions.argsort()[-750:][::-1]

        rec_list[i]=top_pop
        i+=1


    np.save("nlp_toketoppop_rec_list_offline", rec_list)

    eurm = rec_list_to_eurm(rec_list=rec_list)
    eurm = eurm_remove_seed(eurm, dr)

    rec_list = eurm_to_recommendation_list(eurm)

    ev.evaluate(rec_list, "WEILA2_toktoktop_pop",verbose=True, do_plot=True, show_plot=True, save=True, )
