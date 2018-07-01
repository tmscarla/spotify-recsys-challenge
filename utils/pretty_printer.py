"""
EXAMPLE:

from utils.pretty_printer import Pretty_printer


dr = Datareader(mode='offline', verbose=False, only_load=True)
rec_list= np.load("knn_rec_list.npy")

pp = Pretty_printer(datareader=dr)
pp.check_sub(recommendation_list= rec_list  ,target_pids= [33061,33780], topn=10)

"""

from utils.datareader import Datareader
from random import sample
import numpy as np
import scipy.sparse as sps


class Pretty_printer(object):

    def __init__(self, datareader):
        self.dr = datareader

        ###### tracks
        track_df = self.dr.get_df_tracks()

        tids = list(track_df['tid'].as_matrix())
        tr_names = list(track_df['track_name'].as_matrix())

        self.track_id_to_name_dict = dict(zip(tids, tr_names))

        ##### playlists

        playlist_train_df = self.dr.get_df_train_playlists()
        playlist_test_df = self.dr.get_df_test_playlists()

        pl_pid = list(playlist_train_df['pid'].as_matrix())
        pl_pid.extend(list(playlist_test_df['pid'].as_matrix()))
        pl_names = list(playlist_train_df['name'].as_matrix())
        pl_names.extend(list(playlist_test_df['name'].as_matrix()))

        self.playlist_id_to_name_dict = dict(zip(pl_pid, pl_names))

        #### artists
        artist_df =  self.dr.get_df_artists()

        arids = list(artist_df['arid'].as_matrix())
        ar_name = list(artist_df['artist_name'].as_matrix())

        self.artist_id_to_artist_name = dict(zip(arids, ar_name))
        self.track_id_to_artist_id = self.dr.get_track_to_artist_dict()

    def rewrite_file(self, file_path):
        print("#TODO")
        pass

    def __string_one_song(self, song_id, artist=None, num=None):
        to_ret = self.track_id_to_name_dict[song_id]
        if num:
            to_ret ="(" + str(num) + ") "+to_ret
        if artist:
            return to_ret+"{" + self.artist_id_to_artist_name[self.track_id_to_artist_id[song_id]] + "}\t\t"
        return to_ret


    def check_sub(self, recommendation_list, topn=None, artist=True, target_pids=None):

        test_pids_by_cat = self.dr.get_test_pids()

        # indices = self.dr.get_test_pids_indices()

        #for each playlist
        for i in range(len(test_pids_by_cat)):
            if (target_pids is None) or test_pids_by_cat[i] in target_pids  :

                print("PLAYLIST:",i,str(test_pids_by_cat[i]),
                      str(self.playlist_id_to_name_dict[test_pids_by_cat[i]]),"\t\tSONGS:\t", end="")
                #for each track
                for j in range(len(recommendation_list[i])):
                    print(self.__string_one_song(recommendation_list[i][j], artist=artist, num=j+1), end="")

                    if topn and j+1 >= topn:
                        break
                print()


if __name__ == '__main__':

    dr_online = Datareader(mode='online', verbose=False, test_num=1, only_load=True)
    pp = Pretty_printer(datareader=dr_online)

    eurm= sps.load_npz('/Users/lele/Desktop/spotifai/Spotify-Challenge/ensembled_jess2_online.npz')
    from utils.post_processing import eurm_to_recommendation_list
    rec_list = eurm_to_recommendation_list(eurm)

    pp.check_sub(recommendation_list= rec_list ,topn=10)


class PrettyPrint:
    def __init__(self,dr):
        ##### playlists

        playlist_train_df = dr.get_df_train_playlists()
        playlist_test_df = dr.get_df_test_playlists()

        pl_pid = list(playlist_train_df['pid'].as_matrix())
        pl_pid.extend(list(playlist_test_df['pid'].as_matrix()))
        pl_names = list(playlist_train_df['name'].as_matrix())
        pl_names.extend(list(playlist_test_df['name'].as_matrix()))

        self.playlist_id_to_name_dict = dict(zip(pl_pid, pl_names))

        #### albums
        album_df =  dr.get_df_albums()

        alids = list(album_df['alid'].as_matrix())
        al_name = list(album_df['album_name'].as_matrix())

        self.album_id_to_album_name = dict(zip(alids, al_name))
        self.track_id_to_album_id = dr.get_track_to_album_dict()

        #### artists
        artist_df =  dr.get_df_artists()

        arids = list(artist_df['arid'].as_matrix())
        ar_name = list(artist_df['artist_name'].as_matrix())

        self.artist_id_to_artist_name = dict(zip(arids, ar_name))
        self.track_id_to_artist_id = dr.get_track_to_artist_dict()

        ###### tracks
        track_df = dr.get_df_tracks()

        tids = list(track_df['tid'].as_matrix())
        tr_names = list(track_df['track_name'].as_matrix())

        self.track_id_to_name_dict = dict(zip(tids, tr_names))
    
    def printInfoTrack(self,tid):
        print(self.artist_id_to_artist_name[self.track_id_to_artist_id[tid]]
              + '\t---> ' +self.album_id_to_album_name[self.track_id_to_album_id[tid]]  
              + '\t---> ' +self.track_id_to_name_dict[tid])
