from tqdm import tqdm
import numpy as np
import scipy.sparse as sps
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from fim import apriori, fpgrowth, fim, arules
from utils.evaluator import Evaluator
from utils.datareader import Datareader

from personal.Lele.prefixspan import PrefixSpan
from utils.post_processing import rec_list_to_eurm,eurm_to_recommendation_list,eurm_remove_seed


def not_empty_lines_by_target(urm_pos, target_list, min_songs_in_common):
    tmp = urm_pos.tocsc(copy=True)
    tmp = tmp[:, target_list]
    tmp = tmp.tocsr()
    not_empty_lines = list()
    for i in range(tmp.shape[0]):
        if len(tmp.indices[tmp.indptr[i]:tmp.indptr[i + 1]]) >= min_songs_in_common :
            not_empty_lines.append(i)
    return not_empty_lines



def urm_to_sequences(urm_pos, target_list, min_common, list_of_list_of_listss=False):
    sequences_spm = []

    not_empty_lines = not_empty_lines_by_target(urm_pos, target_list, min_common)
    filtered = urm_pos[not_empty_lines]
    for row in tqdm((range(filtered.shape[0])), desc='Converting eurm into list of lists'):
        to_append = list(filtered.indices[filtered.indptr[row]:filtered.indptr[row + 1]]
                             [np.argsort(filtered.data[filtered.indptr[row]:filtered.indptr[row + 1]])])
        if list_of_list_of_listss:
            sequences_spm.append( [[i] for i in to_append])
        else:
            sequences_spm.append(to_append)
    return sequences_spm


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


verbose = False

if __name__ == "__main__":


    dr = Datareader(mode='offline', train_format='50k', verbose=False, only_load=True)
    ev = Evaluator(dr)

    test_known_tracks = build_test_dict(dr)

    test_pids_cat2 = dr.get_test_pids(cat=2)

    rec_list = np.zeros(shape=(10000,500))
    pred = np.zeros(shape=(10000, 2262292))

    for i in tqdm(range(1000,2000)):

        # print("prima target")
        # print(test_pids_cat2[0])
        # print(test_known_tracks[test_pids_cat2[0]])
        # print([x[1] for x in test_known_tracks[test_pids_cat2[0]]])
        #
        # print("start")
        sequences = urm_to_sequences(urm_pos=dr.get_position_matrix(position_type='last'),
                                     target_list=[x[1] for x in test_known_tracks[test_pids_cat2[0]]],
                                     min_common=1)


        # for s in sequences: print(s)
        # for s in sequences[0:2]:
        #     print("seuences:", s)

        # print("maximal")
        seq = fim(sequences[0:2], target='maximal', supp=-2, zmin=2, report='a')
        # for s in seq:
        #     print("max>", s)

        # print("normale")
        # seq = fim(sequences[0:10],  supp=-2, zmin=2, report='a')
        # print("norm", seq)

        # print("prefixspan")
        sequences_for_prefix = urm_to_sequences(urm_pos=dr.get_position_matrix(position_type='last'),
                                             target_list=[x[1] for x in test_known_tracks[test_pids_cat2[0]]],
                                            min_common=1,
                                            list_of_list_of_listss=True)

        model = PrefixSpan.train(sequences_for_prefix, minSupport=0.1, maxPatternLength=250)
        result = model.freqSequences().collect()


        result_dict= dict()

        for fs in result:

            for song in fs.sequence:

                if song in result_dict:

                    result_dict[song] = result_dict[song]+fs.freq*len(fs.sequence)

                else:
                    result_dict[song] = fs.freq


        for song_predicted in result_dict:

            pred[i,song_predicted] = result_dict[song_predicted]





    eurm = eurm_remove_seed(pred , dr )

    rec_list = eurm_to_recommendation_list(eurm)

    ev.evaluate(rec_list, "cat2_top",verbose=True, do_plot=True, show_plot=True, save=True, )



# seuences: [15565, 6186, 6288, 6292, 6294, 6295, 6298, 6310, 6334, 6336, 6337, 6339, 6340, 6362, 6380, 6387, 7597, 7603, 7604, 7605, 7606, 7607, 6173, 6077, 6040, 6027, 74, 76, 77, 81, 282, 768, 2163, 2506, 2507, 2508, 7609, 3084, 3166, 3183, 3282, 3283, 3697, 4211, 4420, 4443, 4493, 6019, 3162, 73, 8408, 8460, 15544, 15545, 15546, 15547, 15548, 15549, 15550, 15551, 15552, 15553, 15554, 15555, 15556, 15557, 15558, 15559, 15560, 15561, 15562, 15563, 15564, 15543, 15503, 15152, 14809, 8484, 8940, 10480, 10527, 10820, 11192, 11200, 11482, 11500, 11512, 8409, 12605, 12710, 12714, 12716, 12728, 12794, 13689, 13692, 14467, 14797, 14801, 12610, 51]
# seuences: [11500]
#
#
#
# [[11500], [12714]], 62
# [[11500], [70]], 62
# [[11500], [64]], 70
# [[11500], [14809]], 71
# [[11500], [13893]], 72
# [[11500], [69]], 81
# [[11500], [69], [68]], 46
# [[11500], [68]], 88
# [[11500], [81]], 101
# [[11500], [81], [68]], 44
