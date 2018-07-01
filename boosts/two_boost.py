import sys
from scipy import sparse
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from utils.ensembler import *
from recommenders.similarity.s_plus import *
from utils.evaluator import Evaluator
from tqdm import tqdm
import random


def two_boost(rec_list, datareader, sim_al, sim_ar,
              top_al=3, top_ar=3, prob=[0.7, 0.2, 0.1]):
    """
    Subsitute the second category of the rec_list with the combination of different recommendations in
    a round robin fashion.
    :param rec_list: the original recommendation list
    :param datareader: a Datareader object
    :param top_al: top similar albums to consider
    :param top_ar: top similar artists to consider
    :param prob: list of probabilities to choose a prediction from [original, albums, artists]
    :return: rec_list_new: the original rec_list with new predictions for category 2
    """

    # Test playlists
    test_pids = dr.get_test_pids(cat=2)

    # Popularity
    urm = dr.get_urm()
    tracks_pop = np.argsort(urm.sum(axis=0).A1)[::-1]

    # A list of known tracks for each test playlist
    known_tracks = datareader.get_df_test_interactions().groupby(['pid'])['tid'].apply(list)

    # Dictionaries
    track_to_artist = datareader.get_track_to_artist_dict()
    track_to_album = datareader.get_track_to_album_dict()

    if datareader.online():
        no_check_duplicates = False
        album_to_tracks = load_obj(name='album_tracks_dict_online', path=ROOT_DIR + '/boosts/')
        artist_to_tracks = load_obj(name='artist_tracks_dict_online', path=ROOT_DIR + '/boosts/')
    else:
        no_check_duplicates = True
        album_to_tracks = load_obj(name='album_tracks_dict_offline', path=ROOT_DIR + '/boosts/')
        artist_to_tracks = load_obj(name='artist_tracks_dict_offline', path=ROOT_DIR + '/boosts/')

    # Initialize rec list for album and artist
    rec_list_album = []
    rec_list_artists = []

    # Albums
    for i in range(1000):

        alid = track_to_album[known_tracks[test_pids[i]][0]]
        top_val = np.argsort(sim_al.data[sim_al.indptr[alid]: sim_al.indptr[alid+1]])[::-1][:top_al]

        top_alb = sim_al.indices[sim_al.indptr[alid]: sim_al.indptr[alid+1]][top_val]

        # Tracks from top similar albums ordered by popularity
        alb_tracks = []
        for a in top_alb:
            alb_tracks.extend(album_to_tracks[a])

        rec_list_album.append(tracks_pop[alb_tracks])

    # Artists
    for i in range(1000):

        arid = track_to_artist[known_tracks[test_pids[i]][0]]
        top_val = np.argsort(sim_ar.data[sim_ar.indptr[arid]: sim_ar.indptr[arid+1]])[::-1][:top_ar]

        top_art = sim_ar.indices[sim_ar.indptr[arid]: sim_ar.indptr[arid+1]][top_val]

        # Tracks from top similar artists ordered by popularity
        art_tracks = []
        for a in top_art:
            art_tracks.extend(artist_to_tracks[a])

        rec_list_artists.append(tracks_pop[art_tracks])

    # Fill the rec_list in a probabilistic way
    rec_list_new = rec_list.copy()
    no_pred = False

    for i in tqdm(range(1000), desc='Fill rec_list'):
        # Re-initialize rec_list_new for category 2
        rec_list_new[i + 1000] = []
        prediction = []

        # Indices
        i_rec = 0
        i_al = 0
        i_ar = 0

        # Iterate until a row is not full
        while len(prediction) < 500:
            p = random.uniform(0, 1)

            # Pick from original rec_list
            if p < prob[0]:
                if i_rec < len(rec_list[i+1000]):
                    if rec_list[i+1000][i_rec] not in prediction or no_check_duplicates:
                        prediction.append(rec_list[i + 1000][i_rec])
                i_rec += 1

            # Pick from album rec_list
            elif p >= prob[0] and p < prob[0] + prob[1]:
                if i_al < len(rec_list_album[i]):
                    if rec_list_album[i][i_al] not in prediction or no_check_duplicates:
                        prediction.append(rec_list_album[i][i_al])
                i_al += 1

            # Pick from artist rec_list
            else:
                if i_ar < len(rec_list_artists[i]):
                    if rec_list_artists[i][i_ar] not in prediction or no_check_duplicates:
                        prediction.append(rec_list_artists[i][i_ar])
                i_ar += 1

            # If no more predictions available from the three lists
            if i_rec >= len(rec_list[i+1000]):
                no_pred = True
                print('WARNING: no predcitions available for row ' + str(i + 1000) + '. It will be filled'
                      'with top pop tracks.')

            # Fill with top-pop
            if no_pred:
                k = 0
                while len(prediction) < 500:
                    if tracks_pop[k] not in prediction:
                        prediction.append(tracks_pop[k])
                    k += 1

        # Substitute new round-robin prediction to rec_list
        rec_list_new[i + 1000] = prediction

    return rec_list_new


if __name__ == '__main__':

    # INITIALIZATION
    dr = Datareader(mode='offline', only_load=True, verbose=False)
    ev = Evaluator(dr)
    test_pids = dr.get_test_pids()

    # PARAMS
    knn = 500
    topk = 750

    # LOAD EURM
    eurm = sparse.load_npz(ROOT_DIR + '/data/ensemble_per_cat_offline_new_data_32_maggio.npz')
    rec_list = eurm_to_recommendation_list(eurm, datareader=dr)

    # SIMILARITIES
    # ucm_album = dr.get_ucm_albums()
    # sim_album = tversky_similarity(ucm_album.T, ucm_album, shrink=200,
    #                                alpha=0.1, beta=1, k=knn, verbose=1, binary=False)
    # sim_album = sim_album.tocsr()
    sim_album = sparse.load_npz(ROOT_DIR + '/data/sim_album.npz')

    # ucm_artist = dr.get_ucm_artists()
    # sim_artist = tversky_similarity(ucm_artist.T, ucm_artist, shrink=200,
    #                                alpha=0.1, beta=1, k=knn, verbose=1, binary=False)
    # sim_artist = sim_artist.tocsr()
    sim_artist = sparse.load_npz(ROOT_DIR + '/data/sim_artist.npz')

    # TWOBOOST
    rec_list_new = two_boost(rec_list, dr, sim_al=sim_album, sim_ar=sim_artist, prob=[0.85, 0.1, 0.05])

    # EVALUATION
    ev.evaluate(rec_list_new, name='toptwo')
