"""

from utils.datareader import Datareader

dr = Datareader(mode='offline', only_load=True)

"""
import pandas as pd
import numpy as np
import os
import gc
import pickle
import scipy.sparse as sp
import ast
from utils.definitions import ROOT_DIR
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

MAX_TESTS = 10
NUM_TRACKS = 2262292
NUM_PLAYLISTS = 1000000
NUM_PLAYLISTS_FULL = 1049361
NUM_PLAYLISTS_TEST = 10000


class Datareader(object):

    __dtype_str = str
    __separator = '\t'
    __dtype_int = np.int32
    __dtype_bool = np.bool_

    __albums_file = 'albums.csv'
    __tracks_file = 'tracks.csv'
    __artists_file = 'artists.csv'
    __train_playlists_file = 'train_playlists.csv'
    __test_interactions_file = 'test_interactions.csv'

    __artists_improved_file = 'artists_improved.csv'
    __tracks_improved_file = 'tracks_improved.csv'

    __path_original_csv = ROOT_DIR+'/data/original/'
    
    ## For creative track
    __path_enriched_icm = ROOT_DIR + '/data/enriched/tmp_icms/'
    ##

    def __init__(self, mode, only_load, train_format='', test_num='1', verbose=True, type="new"):
        """
        :param mode:            "online" or "offline
        :param only_load:       if True, it will try to load already instanciated urms

        :param train_format:    "" for full or "50k" or "100k" or "200k" or "400"
        :param test_num:        "0" or "1" for now
        :param verbose:         True by default
        """

        assert mode=='online' or mode=='offline'
        assert int(test_num)>=0 and int(test_num)<=MAX_TESTS , ' test num ['+str(test_num)+'] not present'
        assert train_format=='' or train_format=='50k' or train_format=='100k' or train_format=='200k' or train_format=='400k'

        if type == "new":
            self.__test_playlist_file = 'test_playlists.csv'
        if type == "old":
            self.__test_playlist_file = 'test_playlists_old.csv'
        self.__verbose = verbose
        self.__only_load = only_load
        self.__train_format = train_format
        self.__online_or_offline = mode
        self.__test_num = test_num

        if self.__online():
            assert len(train_format)==0, 'only for offline you can change trainformat'

        if verbose: print('IMPORTANT: all the returned matrices are in CSR format!!!')
        if verbose and self.__offline(): print('[working on: TEST_NUM: '+str(test_num)+' ]')
        if verbose and self.__online(): print('[working on: ONLINE ]')

        if self.__offline():
            self.__path =  ROOT_DIR + '/data/test'+str(test_num)+'/'
            self.__eval_interactions_file = 'eval_interactions.csv'
            if len(self.__train_format)>0:
                self.__train_interactions_file = 'train_interactions_'+train_format+'.csv'
            else:
                self.__train_interactions_file = 'train_interactions.csv'
        if self.__online():
            self.__path = self.__path_original_csv
            self.__train_interactions_file = 'train_interactions.csv'

        self.__path_matrices = self.__path + 'matrices/'


    def get_urm_without_challengeset(self):
        assert self.online() , "GET_URM_WITHOUT_CHALLENGESET IS USABLE ONLY FOR ONLINE"

        urm = self.get_urm()
        urm.data[urm.indptr[1000000]:] = 0
        urm.eliminate_zeros()

        return urm

    def get_urm(self, binary=False, save_on_disk= False):
        """
        :param save_on_disk:    saves the urm. if the datareader is "only load" mode, this will be set to True.
        :return:                urm matrix. sparse CSR format
                                SHAPE online   (1049361, 2262292)
                                SHAPE offline  (1000000, 2262292)
        """

        urm_name = self.__build_name(name='urm', format=self.__train_format)

        if self.__only_load:
            file_full_path = self.__path_matrices + urm_name + '.npz'
            if os.path.isfile(file_full_path):
                if self.__verbose:print("[URM " + urm_name + ".npz loaded from "+file_full_path+" ]")

                urm = self.__load_matrix(urm_name)
                if binary:
                    urm.data = np.ones(len(urm.data))
                return urm
            else:
                if self.__verbose: print("[URM "+urm_name+".npz never instanciated, now CREATING IT ]")
                save_on_disk=True
        # Dataframe with interactions
        df_train = self.get_df_train_interactions()
        df_test = self.get_df_test_interactions()
        df = pd.concat([df_train, df_test], axis=0, join='outer') # union of the train and test

        # collect data to build urm
        playlists = df['pid'].values
        tracks = df['tid'].values
        assert (playlists.size == tracks.size)
        if self.__online():
            n_playlists = NUM_PLAYLISTS_FULL
        else:
            n_playlists = NUM_PLAYLISTS
        n_tracks = NUM_TRACKS
        n_interactions = tracks.size

        # building the urm
        urm = sp.csr_matrix((np.ones(n_interactions), (playlists,tracks)), shape=(n_playlists, n_tracks),
                            dtype=self.__dtype_int)

        if self.__verbose: print('URM created (%dx%d) - %d interactions' % (n_playlists, n_tracks, n_interactions))

        # save on disk
        if save_on_disk:
            self.__save_matrix(urm_name,urm)

        if binary:
            urm.data = np.ones(len(urm.data))
        return urm

    def get_urm_shrinked(self, save_on_disk= False):
        """
        :param save_on_disk:  if true, matrix and dictionaries are saved in matrices folder
        :return:    urm matrix, dictionary_normal_to_shrinked, dictionary_shrinked_to_normal
                    SHAPE online        (1010000,2262292)
                    SHAPE offline  full ( 1kk   ,2262292)
                    SHAPE offline  50k  ( 60k   ,2262292)
                    SHAPE offline  100k ( 110k  ,2262292)
                    SHAPE offline  200k ( 210k  ,2262292)
                    SHAPE offline  400k ( 410k  ,2262292)
                    10k are the test interactions that are always part of the matrix
        """
        assert (len(self.__train_format)>0 and self.__offline())or self.__online(),"cannot shrink that"
        urm_name = self.__build_name(name='urm', format=self.__train_format, to_add='shrinked')
        dict_n_name = self.__build_name(name='dict_normal_to_shrinked', format=self.__train_format)
        dict_s_name = self.__build_name(name='dict_shrinked_to_normal', format=self.__train_format)

        if self.__only_load:
            file_full_path = self.__path_matrices + urm_name+'.npz'
            if os.path.isfile(file_full_path):
                if self.__verbose: print("[URM " + urm_name + " loaded from "+file_full_path+" ]")
                return sp.load_npz(self.__path_matrices + urm_name+'.npz').tocsr(), \
                       self.__load_dictionary(dict_n_name), \
                       self.__load_dictionary(dict_s_name)
            else:
                if self.__verbose: print("[URM " + urm_name + ".npz never instanciated, now CREATING IT ]")
                save_on_disk = True
        # Dataframe with interactions
        df_train = self.get_df_train_interactions()
        df_test = self.get_df_test_interactions()
        df = pd.concat([df_train, df_test], axis=0, join='outer')  # union of the train and test

        # collect data to build urm
        playlists = df['pid'].values
        tracks = df['tid'].values
        assert (playlists.size == tracks.size)
        if self.__online():
            n_playlists = NUM_PLAYLISTS_FULL
        else:
            n_playlists = NUM_PLAYLISTS
        n_tracks = NUM_TRACKS
        n_interactions = tracks.size

        # building the urm
        urm = sp.csr_matrix((np.ones(n_interactions), (playlists, tracks)), shape=(n_playlists, n_tracks),
                            dtype=self.__dtype_int)

        #concatenates the pids of test and train
        pids_train = np.sort(np.unique(df_train['pid'].values))

        pids_test = self.get_test_pids()
        pids = list(np.sort( np.r_[pids_train,pids_test] ))

        #shrinked urm based on train_pids and test_pids
        urm_shrinked = urm[pids]
        if self.__verbose: print('URM_SHRINKED created',urm_shrinked.shape,'-',np.sum(urm_shrinked.data),'interactions')

        dict_pl_normal_to_shrinked = dict(zip(pids, list(np.arange(len(pids)))))
        dict_pl_shrinked_to_normal = dict(zip(list(np.arange(len(pids))), pids))

        # save on disk
        if save_on_disk:
            self.__save_matrix(urm_name, urm_shrinked)
            self.__save_dictionary(dict_n_name, dict_pl_normal_to_shrinked)
            self.__save_dictionary(dict_s_name, dict_pl_shrinked_to_normal)
        del(urm, df,df_train,df_test)
        gc.collect()

        return urm_shrinked, dict_pl_normal_to_shrinked, dict_pl_shrinked_to_normal


    def get_position_matrix(self, position_type, only_mpd=True):
        """
        :param position_type:       'first' or 'last'
        :return:                Position 'urm' matrix. sparse CSR format
                                it'a an urm where instad of having the 0/1 values, you have the position of
                                the track inside the playlist.
                                if there is a duplicate, the tie breaker is the  first/last rule decided

                                SHAPE online   (1049361, 2262292)
                                SHAPE offline  (1000000, 2262292)


        """

        urm_name = self.__build_name(name='urm_pos', format=self.__train_format, to_add=position_type)
        if self.__only_load:
            file_full_path = self.__path_matrices + urm_name + '.npz'
            if os.path.isfile(file_full_path):
                if self.__verbose:print("[URM " + urm_name + ".npz loaded from "+file_full_path+" ]")
                return self.__load_matrix(urm_name)
            else:
                if self.__verbose: print("[URM "+urm_name+".npz never instanciated, now CREATING IT ]")
                save_on_disk=True

        df_train = self.get_df_train_interactions()
        df_test = self.get_df_test_interactions()

        df = pd.concat([df_train, df_test], axis=0, join='outer') # union of the train and test

        if position_type=='first':
            if self.__verbose: print("[getting first positions]")
            df = df.groupby(['tid', 'pid'], as_index=False )['pos'].min()
        elif position_type=='last':
            if self.__verbose: print("[getting last positions]")
            df = df.groupby(['tid', 'pid'], as_index=False )['pos'].max()
        else:
            print("wrong position, only last or first ["+position_type+"]")

        # collect data to build urm
        pos = df['pos'].values+1
        playlists = df['pid'].values
        tracks = df['tid'].values
        if self.__online():
            n_playlists = NUM_PLAYLISTS_FULL
        else:
            n_playlists = NUM_PLAYLISTS

        n_tracks = NUM_TRACKS
        assert (playlists.size == tracks.size)

        position_matrix = sp.csr_matrix((pos, (playlists, tracks)), shape=(n_playlists, n_tracks),
                            dtype=self.__dtype_int)

        if self.__verbose: print('position matrix created (%dx%d) ' % (n_playlists, n_tracks))

        if save_on_disk:
            self.__save_matrix(urm_name, position_matrix)

        return position_matrix

    def get_urm_with_position(self, n_adjacents):
        """
        Return an augmented URM (1M x 567M) which takes into account positions of each track.
        :param n_adjacents: number of eventually adjacents position to take into account
        :return: urm: the augmented urm
        """
        urm = self.get_urm(binary=True)
        pos_matrix = self.get_position_matrix(position_type='last')

        rows = []
        cols = []
        data = []

        for p in tqdm(range(pos_matrix.shape[0])):
            start = pos_matrix.indptr[p]
            end = pos_matrix.indptr[p + 1]

            tracks = pos_matrix.indices[start:end]
            positions = pos_matrix.indices[start:end]

            for idx in range(len(tracks)):
                if positions[idx] <= 250:
                    effective_pos = (tracks[idx] * positions[idx]) + tracks[idx]

                    rows.append(p)
                    cols.append(effective_pos)
                    data.append(1)

                    # Consider adjacent positions
                    if n_adjacents > 0:
                        step = 1.0 / (n_adjacents + 1)

                        for a in range(n_adjacents):
                            value = (n_adjacents - a) * step

                            # Before
                            if effective_pos - (a + 1) >= 0:
                                rows.append(p)
                                cols.append(effective_pos - (a + 1))
                                data.append(value)
                            # After
                            if effective_pos + (a + 1) < 250:
                                rows.append(p)
                                cols.append(effective_pos + (a + 1))
                                data.append(value)

        urm_pos_ext = sp.csr_matrix((data, (rows, cols)), shape=(pos_matrix.shape[0], 250 * pos_matrix.shape[1]))

        new_urm = sp.hstack((urm, urm_pos_ext))

        return new_urm

    def get_urm_with_pop_clusters(self, n_clusters):
        urm = self.get_urm()
        urm_T = (urm.T).tocsr()

        pop = urm.sum(axis=0).A1
        pop_idx = np.argsort(pop)[::-1]

        tracks_per_cluster = int(urm.shape[1] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), desc='URM popularity'):
            # If last iteration
            if i == n_clusters - 1:
                tracks = pop_idx[tracks_per_cluster * i:]
            else:
                tracks = pop_idx[tracks_per_cluster * i: tracks_per_cluster * (i + 1)]

            rows.extend(tracks)
            cols.extend([i for x in range(len(tracks))])
            data.extend([1 for x in range(len(tracks))])

    def get_urm_test(self, save_on_disk=False):
        urm_name = self.__build_name(name='urm_testonly')
        dict_n_name = self.__build_name(name='dict_testonly_normal_to_shrinked', format=self.__train_format)
        dict_s_name = self.__build_name(name='dict_testonly_shrinked_to_normal', format=self.__train_format)

        if self.__only_load:
            file_full_path = self.__path_matrices + urm_name + '.npz'
            if os.path.isfile(file_full_path):
                if self.__verbose: print("[URM " + urm_name + " loaded from " + file_full_path + " ]")
                return sp.load_npz(file_full_path).tocsr(), \
                       self.__load_dictionary(dict_n_name), \
                       self.__load_dictionary(dict_s_name)
            else:
                if self.__verbose: print("[URM " + urm_name + ".npz never instanciated, now CREATING IT ]")
                save_on_disk = True
        # Dataframe with interactions
        df = self.get_df_test_interactions()

        playlists = df['pid'].values
        tracks = df['tid'].values
        pids = self.get_test_pids()

        urm = sp.csr_matrix((np.ones(tracks.size), (playlists, tracks)), shape=(NUM_PLAYLISTS_FULL, NUM_TRACKS),
                                dtype=self.__dtype_int)
        print(urm.shape, urm.nnz)
        urm_shrinked = urm[pids]
        print(urm_shrinked.shape, urm.nnz)
        if self.__verbose: print('URM_SHRINKED created', urm_shrinked.shape, '-', np.sum(urm_shrinked.data),
                                 'interactions')
        dict_pl_normal_to_shrinked = dict(zip(pids, list(np.arange(len(pids)))))
        dict_pl_shrinked_to_normal = dict(zip(list(np.arange(len(pids))), pids))
        # save on disk
        if save_on_disk:
            self.__save_matrix(urm_name, urm_shrinked)
            self.__save_dictionary(dict_n_name, dict_pl_normal_to_shrinked)
            self.__save_dictionary(dict_s_name, dict_pl_shrinked_to_normal)
        del (urm, df)
        gc.collect()
        return urm_shrinked, dict_pl_normal_to_shrinked, dict_pl_shrinked_to_normal

    def get_evaluation_urm(self):
        df = self.get_df_eval_interactions()

        playlists = df['pid'].values
        tracks = df['tid'].values
        if self.__online():
            n_playlists = NUM_PLAYLISTS_FULL
        else:
            n_playlists = NUM_PLAYLISTS
        n_tracks = NUM_TRACKS
        n_interactions = tracks.size

        eval_urm = sp.csr_matrix((np.ones(n_interactions), (playlists,tracks)), shape=(n_playlists, n_tracks),
                            dtype=self.__dtype_int)

        if self.__verbose: print('EVALUATION URM created (%dx%d) - %d interactions' % (n_playlists, n_tracks, n_interactions))
        return eval_urm

    def get_icm(self, arid=False, alid=False):
        """
        :param arid:        if true, it will be added in the icm
        :param alid:        if true, it will be added in the icm
        :param last_seen:   TODO
        :param first_seen:  TODO
        :return:    sparse csr matrix of shape  (UNIQUE_TRACKS, UNIQUE_ARTISTS+ UNIQUE_ALBUMS)
                                                (2262292      , 295860        + 734684       )
        """
        # if no feature selected
        if not arid and not alid:
            raise ValueError('ERROR: no feature selected in ICM!!!')

        if self.__verbose: print("[reading ICM from "+self.__path_original_csv+self.__tracks_file+" ]", end=" ")
        df_tracks = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator, header=0,
                         usecols=['tid', 'arid', 'alid'],
                         dtype={'tid': self.__dtype_int, 'arid': self.__dtype_int, 'alid': self.__dtype_int})
        if self.__verbose: print('>DF tracks read')

        # start building icm
        n_tracks = df_tracks['tid'].max() + 1  # index starts from 0
        tracks = df_tracks['tid'].values

        icm = sp.csr_matrix((n_tracks, 0)) # empty dummy, and stacking features on it
        if arid:
            artists = df_tracks['arid'].values
            n_artists = artists.max() + 1  # index starts from 0
            # create partial icm artists
            icm_ar = sp.csr_matrix((np.ones(n_tracks), (tracks, artists)), shape=(n_tracks, n_artists),
                                   dtype=self.__dtype_int)
            icm = sp.hstack([icm, icm_ar])
            if self.__verbose: print('ICM artists created \t(%dx%d)' % (icm_ar.shape))
        if alid:
            albums = df_tracks['alid'].values
            n_albums = albums.max() + 1  # index starts from 0
            # create partial icm artists
            icm_al = sp.csr_matrix((np.ones(n_tracks), (tracks, albums)), shape=(n_tracks, n_albums),
                                   dtype=self.__dtype_int)
            icm = sp.hstack([icm,icm_al])
            if self.__verbose: print('ICM albums created \t(%dx%d)' % (icm_al.shape))

        if self.__verbose: print('ICM total created \t(%dx%d)' % (icm.shape))

        return icm.tocsr()

    def get_icm_duration(self, n_clusters):
        """
        Create an ICM (tracks, n_clusters) gathering tracks according to their duration in milliseconds.
        Each cluster has the same number of tracks.
        :param n_clusters: number of clusters
        :return: icm: the item content matrix
        """

        urm = self.get_urm()

        tracks_df = self.get_df_tracks()
        durations = tracks_df['duration_ms'].values

        dur_idx = np.argsort(durations)[::-1]

        tracks_per_cluster = int(urm.shape[1] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), 'ICM duration:'):

            # # If last iteration
            if i == n_clusters - 1:
                tracks = dur_idx[tracks_per_cluster * i:]
            else:
                tracks = dur_idx[tracks_per_cluster * i: tracks_per_cluster * (i + 1)]

            rows.extend(tracks)
            cols.extend([i for x in range(len(tracks))])
            data.extend([1 for x in range(len(tracks))])

        icm = sp.csr_matrix((data, (rows, cols)), shape=(urm.shape[1], n_clusters))

        return icm

    def get_icm_popularity(self, n_clusters, n_adjacents=0):
        """
        Create an ICM (tracks, n_clusters) gathering tracks according to their duration in milliseconds.
        Each cluster has the same number of tracks.
        :param n_clusters: number of clusters
        :param n_adjacents: number of adjacents clusters to be considered
        :return: icm: item content matrix
        """

        urm = self.get_urm()

        pop = urm.sum(axis=0).A1
        pop_idx = np.argsort(pop)[::-1]

        tracks_per_cluster = int(urm.shape[1] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), desc='ICM popularity'):
            # If last iteration
            if i == n_clusters - 1:
                tracks = pop_idx[tracks_per_cluster * i:]
            else:
                tracks = pop_idx[tracks_per_cluster * i: tracks_per_cluster * (i + 1)]

            rows.extend(tracks)
            cols.extend([i for x in range(len(tracks))])
            data.extend([1 for x in range(len(tracks))])

            # Consider adjacent clusters
            if n_adjacents > 0:
                step = 1.0 / (n_adjacents + 1)

                for a in range(n_adjacents):
                    value = (n_adjacents - a) * step

                    # Before
                    if i - (a+1) >= 0:
                        rows.extend(tracks)
                        cols.extend([i-(a+1) for x in range(len(tracks))])
                        data.extend([value for x in range(len(tracks))])
                    # After
                    if i + (a+1) < n_clusters:
                        rows.extend(tracks)
                        cols.extend([i+(a+1) for x in range(len(tracks))])
                        data.extend([value for x in range(len(tracks))])

        icm = sp.csr_matrix((data, (rows, cols)), shape=(urm.shape[1], n_clusters))

        return icm

    def get_ucm_followers(self, n_clusters):
        """
        Create an UCM (playlists, n_clusters) gathering playlists according to their followers.
        Each cluster has the same number of playlists.
        :param n_clusters: number of clusters
        :return: ucm: user content matrix
        """

        urm = self.get_urm()
        train_playlists_df = self.get_df_train_playlists()

        if self.offline():
            test_playlists_df = self.get_df_test_playlists()
            concat_df = pd.concat([train_playlists_df, test_playlists_df])
            concat_df = concat_df.sort_values(['pid'], ascending=True)

            followers = concat_df['num_followers'].values
        else:
            followers = train_playlists_df['num_followers'].values

        followers_idx = np.argsort(followers)[::-1]

        playlists_per_cluster = int(urm.shape[0] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), desc='UCM followers'):
            # If last iteration
            if i == n_clusters - 1:
                playlists = followers_idx[playlists_per_cluster * i:]
            else:
                playlists = followers_idx[playlists_per_cluster * i: playlists_per_cluster * (i + 1)]

            rows.extend(playlists)
            cols.extend([i for x in range(len(playlists))])
            data.extend([1 for x in range(len(playlists))])

        ucm = sp.csr_matrix((data, (rows, cols)), shape=(urm.shape[0], n_clusters))

        return ucm

    def get_eurm_top_pop_first(self, eurm_nlp):
        urm = self.get_urm(binary=True)
        urm_csc = urm.tocsc(copy=True)

        test_pids = self.get_test_pids(cat=1)

        rows = []
        cols = []
        data = []

        for idx in tqdm(range(len(test_pids)), desc='Eurm pop first'):
            start = eurm_nlp.indptr[idx]
            end = eurm_nlp.indptr[idx + 1]

            top_idx = np.argsort(eurm_nlp.data[start:end])[::-1]

            if len(top_idx) > 0:
                top_track = eurm_nlp.indices[start:end][top_idx[0]]

                start_csc = urm_csc.indptr[top_track]
                end_csc = urm_csc.indptr[top_track+1]

                playlists = urm_csc.indices[start_csc:end_csc]

                pop_tracks = urm[playlists].sum(axis=0).A1
                tracks = np.argsort(pop_tracks)[::-1][:500]

                for t in tracks:
                    rows.append(idx)
                    cols.append(t)
                    data.append(pop_tracks[t])
            else:
                pop_all = urm.sum(axis=0).A1
                tracks = np.argsort(pop_all)[::-1][:500]

                for t in tracks:
                    rows.append(idx)
                    cols.append(t)
                    data.append(pop_all[t])


        eurm_top = sp.csr_matrix((data, (rows, cols)), shape=(10000, urm.shape[1]))

        return eurm_top

    def get_eurm_top_pop_filter_cat_1(self, sim_nlp, k, topk):
        """
        Compute a eurm (10K, 2.2M) with top popular tracks just for the fist category. The last 9k lines of the
        matrix are empy. The popularity is based on this criterion:
            - keep the k most similar playlists
            - for each track of playlists find all the playlists that have that track
            - union of all found playlists
            - compute tracks popularity on the union
        :param sim_nlp: a similarity matrix (playlists, playlists) obtained with an nlp algorithm
        :param k: consider the k most similar playlists
        :param topk: cut off for the popularity
        :return: eurm_top: a eurm with popularity for the first category
        """
        print('[ WARNING: be careful that datareader and sim_nlp are both online or offline ]')

        # URM in csr and csc version
        urm = self.get_urm(binary=True)
        urm_csc = urm.tocsc(copy=True)

        test_pids = self.get_test_pids(cat=1)

        rows = []
        cols = []
        data = []

        for idx in tqdm(range(len(test_pids)), desc='Eurm pop cat1'):
            pid = test_pids[idx]

            start = sim_nlp.indptr[pid]
            end = sim_nlp.indptr[pid + 1]

            # Compute k-similar playlists
            top_idx = np.argsort(sim_nlp.data[start:end])[::-1][:k]
            top = sim_nlp.indices[start:end][top_idx]

            tracks = []

            # Gather tracks for each similar playlist
            for p in top:
                start_p = urm.indptr[p]
                end_p = urm.indptr[p + 1]

                tracks.extend(urm.indices[start_p:end_p])

            playlists = []

            # Union of playlists which contain that tracks
            for t in tracks:
                start_t = urm_csc.indptr[t]
                end_t = urm_csc.indptr[t + 1]

                playlists.extend(urm_csc.indices[start_t:end_t])

            playlists = list(set(playlists))

            # Filter pop
            filter_pop = urm[playlists].sum(axis=0).A1
            top_pop_filter = np.argsort(filter_pop)[::-1][:topk]

            rows.extend([idx for x in range(len(top_pop_filter))])
            cols.extend(top_pop_filter)
            data.extend([filter_pop[t] for t in top_pop_filter])

        eurm_top = sp.csr_matrix((data, (rows, cols)), shape=(10000, urm.shape[1]))

        return eurm_top

    def get_icm_improved(self, arid=False, alid=False, weight_main_artist=1, weight_co_artist=1):
        """
        :param arid:        if true, it will be added in the icm
        :param alid:        if true, it will be added in the icm
        :return:    sparse csr matrix of shape  (UNIQUE_TRACKS, UNIQUE_ARTISTS+ UNIQUE_ALBUMS)
                                                (2262292      , 310559        + 734684       )
        """
        # if no feature selected
        if not arid and not alid:
            raise ValueError('ERROR: no feature selected in ICM!!!')

        if self.__verbose: print("[reading ICM IMPROVED from "+self.__path_original_csv+self.__tracks_improved_file+" ]", end=" ")
        df_tracks = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_improved_file, sep=self.__separator, header=0,
                         usecols=['tid', 'new_main_arids', 'new_co_arids', 'alid'],
                         dtype={'tid': self.__dtype_int, 'new_main_arids': 'O', 'new_co_arids': 'O', 'alid': self.__dtype_int})
        if self.__verbose: print('>DF tracks read')

        # start building icm
        n_tracks = df_tracks['tid'].max() + 1  # index starts from 0
        tracks = df_tracks['tid'].values

        icm = sp.csr_matrix((n_tracks, 0)) # empty dummy, and stacking features on it
        if arid:
            if self.__verbose: print("Apply literal eval to arids arrays...")
            main_artists = df_tracks['new_main_arids'].apply(ast.literal_eval).values
            co_artists = df_tracks['new_co_arids'].apply(ast.literal_eval).values
            val, row, col = [], [], []
            for i in tqdm(range(n_tracks)):
                for a in main_artists[i]:
                    row.append(i)
                    col.append(a)
                    val.append(weight_main_artist)
                for a in co_artists[i]:
                    row.append(i)
                    col.append(a)
                    val.append(weight_co_artist)
            # create partial icm artists
            n_artists = max(col)+1
            icm_ar = sp.csr_matrix((val, (row, col)), shape=(n_tracks, n_artists),
                                   dtype=self.__dtype_int)
            icm = sp.hstack([icm, icm_ar])
            if self.__verbose: print('ICM artists created \t(%dx%d)' % (icm_ar.shape))
        if alid:
            albums = df_tracks['alid'].values
            n_albums = albums.max() + 1  # index starts from 0
            # create partial icm artists
            icm_al = sp.csr_matrix((np.ones(n_tracks), (tracks, albums)), shape=(n_tracks, n_albums),
                                   dtype=self.__dtype_int)
            icm = sp.hstack([icm,icm_al])
            if self.__verbose: print('ICM albums created \t(%dx%d)' % (icm_al.shape))

        if self.__verbose: print('ICM total created \t(%dx%d)' % (icm.shape))

        return icm.tocsr()

    def get_ucm_artists(self, verbose=True, remove_duplicates=False):
        """
        Build a ucm (playlists, artists) with artists occurrences as data in the cells.
        If the dataframe is offline: (1M, artists) with playlists ordered by pid
        If the dataframe is online: (1.049M, artists) with 1M train playlists + tests playlists with empy rows!
        :return: ucm: the user content matrix in csr format
        """

        # Dataframes with interactions
        if verbose:
            print('Loading dataframes...')
        df_train = self.get_df_train_interactions()
        df_test = self.get_df_test_interactions()
        df = pd.concat([df_train, df_test], axis=0, join='outer')

        if remove_duplicates:
            df = df[['pid','tid']]
            df.drop_duplicates(subset=None, keep='first', inplace=True)

        playlists = df['pid'].as_matrix()
        tracks = df['tid'].as_matrix()

        dictionary = self.get_track_to_artist_dict()
        artists = [dictionary[t] for t in tracks]

        # Get pids: train + test
        pids = list(self.get_train_pids()) + list(self.get_test_pids())

        # Build UCM
        if verbose:
            print('Building UCM artists...')
        ucm = sp.csr_matrix((np.ones(len(playlists)), (playlists, artists)), shape=(1049361, len(self.get_artists())))
        ucm = ucm.tocsr()

        # Reorder pids if offline
        if self.offline():
            pids = np.sort(pids)
            ucm = ucm[pids]

        return ucm

    def get_ucm_albums(self, verbose=True, remove_duplicates=False):
        """
        Build a ucm (playlists, albums) with albums occurrences as data in the cells.
        If the dataframe is offline: (1M, artists) with playlists ordered by pid
        If the dataframe is online: (1.049M, artists) with 1M train playlists + tests playlists with empy rows!
        :return: ucm: the user content matrix in csr format
        """

        # Dataframes with interactions
        if verbose:
            print('Loading dataframes...')
        df_train = self.get_df_train_interactions()
        df_test = self.get_df_test_interactions()
        df = pd.concat([df_train, df_test], axis=0, join='outer')

        if remove_duplicates:
            df = df[['pid','tid']]
            df.drop_duplicates(subset=None, keep='first', inplace=True)

        playlists = df['pid'].as_matrix()
        tracks = df['tid'].as_matrix()

        dictionary = self.get_track_to_album_dict()
        albums = [dictionary[t] for t in tracks]

        # Get pids: train + test
        pids = list(self.get_train_pids()) + list(self.get_test_pids())

        # Build UCM
        if verbose:
            print('Building UCM albums...')
        ucm = sp.csr_matrix((np.ones(len(playlists)), (playlists, albums)),
                            shape=(1049361, len(self.get_df_albums()['alid'].as_matrix())))
        ucm = ucm.tocsr()

        # Reorder pids if offline
        if self.offline():
            pids = np.sort(pids)
            ucm = ucm[pids]

        return ucm

    def get_eurm_top_pop(self, top_pop_k=500, remove_duplicates=False, binary=False):
        """
        Build a eurm (10K, 2.2M) with k top pop tracks for each playlist.
        :param top_pop_k: keep the k top pop tracks for each row in the matrix
        :param binary: set all values to 1, otherwise preserve popularity values
        :return: eurm_top_pop:
        """
        urm = self.get_urm()

        if remove_duplicates:
            tracks_pop = np.diff(urm.tocsc().indptr)
        else:
            tracks_pop = np.ravel(urm.sum(axis=0))
        popular_tracks = np.argsort(tracks_pop)[::-1][:top_pop_k]

        data = []
        rows = []
        cols = []

        for p in range(10000):
            for t in popular_tracks:
                rows.append(p)
                cols.append(t)

                if binary:
                    data.append(1)
                else:
                    data.append(tracks_pop[t])

        eurm_top_pop = sp.csr_matrix((data, (rows, cols)), shape=(10000, urm.shape[1]))

        if binary is False:
            print('ATTENTION: Binary = False, the returned eurm is not normalized!')

        return eurm_top_pop

    def get_tracks(self, name=False, duration=False):
        """
        :param name:
        :param duration:
        :return:           numpy ndarray of shape (295860,     2)
                                                  ( playlists, features[arid,name,...])
        """

        if self.__verbose: print("[TRACKS: READING csv from "+self.__path_original_csv+self.__tracks_file+" ]", end = " ")
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator, header=0,
                         usecols=['tid', 'track_name', 'duration_ms'],
                         dtype={'tid': self.__dtype_int, 'track_name': self.__dtype_str, 'duration_ms': self.__dtype_int})
        if self.__verbose: print('>DF tracks read')


        # building info
        t_info = [df['tid'].values]
        if name: t_info.append(df['track_name'].values)
        if duration: t_info.append(df['duration_ms'].values)
        t_info = np.array(t_info).T
        return t_info


    def get_train_playlists(self, name=False, description=False, num_followers=False, num_tracks=False, collaborative=False,
                         num_albums=False, num_artists=False, modified_at=False, num_edits=False, duration_ms=False,
                         albums_rate=False, artists_rate=False, _verbose=True ):
        """
        :param name / description/.... / artists_rate:  if true it will be in the numpy array returned

        :return:            numpy ndarray of shape (1000000, x ) where x is (1-11), based on params
        """
        # name	collaborative	pid	modified_at
        # num_albums	num_tracks	num_followers	num_edits	duration_ms	num_artists	description

        if self.__verbose and _verbose: print("[reading PLAYLISTS from " + self.__path + self.__train_playlists_file + " ]", end=" ")
        df = pd.read_csv(filepath_or_buffer=self.__path + self.__train_playlists_file,
                         sep=self.__separator, header=0,
                         usecols=['pid', 'name', 'description', 'num_followers', 'collaborative','num_tracks',
                                  'num_albums', 'num_artists', 'modified_at', 'num_edits', 'duration_ms'],
                         dtype={'pid': self.__dtype_int, 'name': self.__dtype_str, 'description': self.__dtype_str,
                                'num_followers': self.__dtype_int, 'collaborative': self.__dtype_bool,
                                'num_tracks': self.__dtype_int, 'num_albums': self.__dtype_int,
                                'num_artists': self.__dtype_int, 'modified_at': self.__dtype_int,
                                'num_edits': self.__dtype_int,
                                'duration_ms': self.__dtype_int})
        if self.__verbose: print('DF playlsits read')

        # building info
        i = 0
        p_info = [df['pid'].values]
        order = str(i) + '-pid'
        i += 1
        if name: p_info.append(df['name'].values); order += ', ' + str(i) + '-name'; i += 1
        if description: p_info.append(df['description'].values); order += ', ' + str(i) + '-description'; i += 1
        if num_followers: p_info.append(df['num_followers'].values); order += ', ' + str(i) + '-num_followers'; i += 1
        if num_tracks: p_info.append(df['num_tracks'].values); order += ', ' + str(i) + '-num_tracks'; i += 1
        if collaborative: p_info.append(df['collaborative'].values); order += ', ' + str(i) + '-collaborative'; i += 1
        if num_albums: p_info.append(df['num_albums'].values); order += ', ' + str(i) + '-albums'; i += 1
        if num_artists: p_info.append(df['num_artists'].values); order += ', ' + str(i) + '-num_artists'; i += 1
        if modified_at: p_info.append(df['modified_at'].values); order += ', ' + str(i) + '-modified_at'; i += 1
        if num_edits: p_info.append(df['num_edits'].values); order += ', ' + str(i) + '-num_edits'; i += 1
        if duration_ms: p_info.append(df['duration_ms'].values); order += ', ' + str(i) + '-duration_ms'; i += 1

        if albums_rate: p_info.append(df.num_albums / df.num_tracks) ; order += ', ' + str(i) + '-albums_rate'; i += 1
        if artists_rate: p_info.append(df.num_artists / df.num_tracks) ;order += ', ' + str(i) + '-artists_rate'; i += 1

        p_info = np.array(p_info).T
        if self.__verbose and _verbose: print('Index order: ' + order)
        return p_info


    def get_artists(self, name=False ):
        """
        :param name:            if true, append the list of names
        :return:                numpy ndarray of shape (295860, 2)
                                                       (artists, features[arid,name])
        """
        if self.__verbose: print("[reading ARTISTS from "+self.__path_original_csv+self.__artists_file+" ]", end= ' ')
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__artists_file, sep=self.__separator, header=0,
                         usecols=['arid', 'artist_name'],
                         dtype={'arid': self.__dtype_int, 'artist_name': self.__dtype_str})
        if self.__verbose: print('>DF artist read')

        # building info
        ar_info = [df['arid'].values]
        if name: ar_info.append(df['artist_name'].values)
        ar_info = np.array(ar_info).T
        return ar_info

    def get_artists_improved(self, name=False ):
        """
        :param name:            if true, append the list of names
        :return:                numpy ndarray of shape (310558, 2)
                                                       (artists, features[arid,name])
        """
        if self.__verbose: print("[reading ARTISTS IMPROVED from "+self.__path_original_csv+self.__artists_improved_file+" ]", end= ' ')
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__artists_improved_file, sep=self.__separator, header=0,
                         usecols=['new_arid', 'new_artist_name'],
                         dtype={'new_arid': self.__dtype_int, 'new_artist_name': self.__dtype_str})
        if self.__verbose: print('>DF artist read')

        # building info
        ar_info = [df['new_arid'].values]
        if name: ar_info.append(df['new_artist_name'].values)
        ar_info = np.array(ar_info).T
        return ar_info

    def get_albums(self, name=True):
        """
        :param name:            if true, append the list of names
        :return:                numpy ndarray of shape (295860, 2)
                                                           (albums, features[alid,name])
        """
        # read file
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__albums_file, sep=self.__separator, header=0,
                         usecols=['alid', 'album_name'],
                         dtype={'alid': self.__dtype_int, 'album_name': self.__dtype_str})
        # building info
        al_info = [df['alid'].values]
        if name: al_info.append(df['album_name'].values)
        al_info = np.array(al_info).T
        return al_info,df

    def get_icm_albums_name(self):
        """
        :param name:            if true, append the list of names
        :return:                numpy ndarray of shape (295860, 2)
                                                           (albums, features[alid,name])
        """
        # read file
        df_tracks = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator, header=0,
                         usecols=['tid', 'arid', 'alid'],
                         dtype={'tid': self.__dtype_int, 'arid': self.__dtype_int, 'alid': self.__dtype_int})
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__albums_file, sep=self.__separator, header=0,
                         usecols=['alid', 'album_name'],
                         dtype={'alid': self.__dtype_int, 'album_name': self.__dtype_str})
        n_tracks = df_tracks['tid'].max() + 1  # index starts from 0
        tracks = df_tracks['tid'].values
    
        df_complete = df_tracks.set_index("alid").join(df.set_index("alid"))
        df_complete = df_complete.reset_index()
        album_name = df_complete["album_name"]
        dic = dict(zip(album_name.unique(),range(len(album_name.unique()))))
        album_name = map(lambda x:dic[x],album_name)
        album_name = np.fromiter(album_name,dtype=np.int)
        
        n_albums = album_name.max() + 1  # index starts from 0
        # create partial icm artists
        icm = sp.csr_matrix((np.ones(n_tracks), (tracks, album_name)), shape=(n_tracks, n_albums),
                               dtype=np.float32)
        
        return icm
    
    def get_icm_tracks_name(self):
        """
        :param name:            if true, append the list of names
        :return:                numpy ndarray of shape (295860, 2)
                                                           (albums, features[alid,name])
        """
        # read file
        df_tracks = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator, header=0,
                         usecols=['tid', 'arid', 'alid'],
                         dtype={'tid': self.__dtype_int, 'arid': self.__dtype_int, 'alid': self.__dtype_int})
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator, header=0,
                         usecols=['tid', 'track_name', 'duration_ms'],
                         dtype={'tid': self.__dtype_int, 'track_name': self.__dtype_str, 'duration_ms': self.__dtype_int})
        n_tracks = df_tracks['tid'].max() + 1  # index starts from 0
        tracks = df_tracks['tid'].values
    
        df_complete = df_tracks.set_index("tid").join(df.set_index("tid"))
        df_complete = df_complete.reset_index()
        track_name = df_complete["track_name"]
        dic = dict(zip(track_name.unique(),range(len(track_name.unique()))))
        track_name = map(lambda x:dic[x],track_name)
        track_name = np.fromiter(track_name,dtype=np.int)
        
        n_tracks_name = track_name.max() + 1  # index starts from 0
        # create partial icm artists
        icm = sp.csr_matrix((np.ones(n_tracks), (tracks, track_name)), shape=(n_tracks, n_tracks_name),
                               dtype=np.float32)
        
        return icm


    def get_test_playlists(self,name=False,num_holdouts=False,num_samples=False,num_tracks=False,_verbose=True):
        """
        :param name /.../num_tracks:  if true it will be in the numpy array returned
        :return:            numpy ndarray of shape (10k, 2/3/4/5 )
                                                    (playlists id, features[name...numtracks]
        """

        if self.__verbose and _verbose:print("[PLAYLISTS: READING csv from "+self.__path+self.__test_playlist_file+" ]",end=' ')
        df = pd.read_csv(filepath_or_buffer=self.__path + self.__test_playlist_file,
                         sep=self.__separator, header=0,
                         usecols=['pid', 'name', 'num_holdouts','num_samples','num_tracks'],
                         dtype={'pid': self.__dtype_int, 'name': self.__dtype_str,
                                'num_samples':self.__dtype_int, 'num_holdouts':self.__dtype_int,
                                'num_tracks':self.__dtype_int} )

        if self.__verbose: print('>DF test playlsits read')

        # building info
        i = 0
        p_info = [df['pid'].values]
        order = str(i) + '-pid'
        i += 1
        if name: p_info.append(df['name'].values); order += ', ' + str(i) + '-name'; i += 1
        if num_tracks: p_info.append(df['num_tracks'].values); order += ', ' + str(i) + '-num_tracks'; i += 1
        if num_holdouts: p_info.append(df['num_holdouts'].values); order += ', ' + str(i) + '-num_holdouts'; i += 1
        if num_samples: p_info.append(df['num_samples'].values); order += ', ' + str(i) + '-num_samples'; i += 1
        if num_tracks: p_info.append(df['num_tracks'].values); order += ', ' + str(i) + '-num_tracks'; i += 1

        p_info = np.array(p_info).T
        if self.__verbose and _verbose: print('Index order: ' + order)
        return p_info


    def get_test_pids(self, cat='all'):
        """
        :param cat: ['all', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        :return: list of pids of the test playlists
        """

        if cat == 'all':
            if hasattr(self,'__test_pids'):
                return self.__test_pids
            self.__test_pids = self.get_test_playlists(_verbose=False).transpose()[0]

        elif cat == 1:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 0
            df_masked = test_playlists_df[mask]
            self.__test_pids = df_masked['pid'].values

        elif cat == 2:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 1
            df_masked = test_playlists_df[mask]
            self.__test_pids = df_masked['pid'].values

        elif cat == 3:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 5
            mask_title = test_playlists_df['name'].notnull()
            df_masked = test_playlists_df[mask]
            df_masked = df_masked[mask_title]
            self.__test_pids = df_masked['pid'].values

        elif cat == 4:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 5
            mask_title = test_playlists_df['name'].isnull()
            df_masked = test_playlists_df[mask]
            df_masked = df_masked[mask_title]
            self.__test_pids = df_masked['pid'].values

        elif cat == 5:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 10
            mask_title = test_playlists_df['name'].notnull()
            df_masked = test_playlists_df[mask]
            df_masked = df_masked[mask_title]
            self.__test_pids = df_masked['pid'].values

        elif cat == 6:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 10
            mask_title = test_playlists_df['name'].isnull()
            df_masked = test_playlists_df[mask]
            df_masked = df_masked[mask_title]
            self.__test_pids = df_masked['pid'].values

        elif cat == 7 or cat == 8:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 25
            df_masked = test_playlists_df[mask]
            pids_unfiltered = df_masked['pid'].values

            test_interactions_df = self.get_df_test_interactions()
            mask_unfiltered = test_interactions_df['pid'].isin(pids_unfiltered)
            interactions_df_masked_unflitered = test_interactions_df[mask_unfiltered]
            interactions_df_masked_unflitered.sort_values(['pos'], ascending=True)

            known_tracks = interactions_df_masked_unflitered.groupby(['pid'])[['pos', 'tid']] \
                .apply(lambda x: x.values.tolist())
            for s in known_tracks:
                s = s.sort(key=lambda x: x[0])

            pids_filtered = []
            for p in pids_unfiltered:
                if known_tracks[p][0][0] == 0 and known_tracks[p][-1][0] == 24:
                    if cat == 7:
                        pids_filtered.append(p)
                else:
                    if cat == 8:
                        pids_filtered.append(p)

            self.__test_pids = pids_filtered

        elif cat == 9 or cat == 10:
            test_playlists_df = self.get_df_test_playlists()
            mask = test_playlists_df['num_samples'] == 100
            df_masked = test_playlists_df[mask]
            pids_unfiltered = df_masked['pid'].values

            test_interactions_df = self.get_df_test_interactions()
            mask_unfiltered = test_interactions_df['pid'].isin(pids_unfiltered)
            interactions_df_masked_unflitered = test_interactions_df[mask_unfiltered]
            interactions_df_masked_unflitered.sort_values(['pos'], ascending=True)

            known_tracks = interactions_df_masked_unflitered.groupby(['pid'])[['pos', 'tid']] \
                .apply(lambda x: x.values.tolist())
            for s in known_tracks:
                s = s.sort(key=lambda x: x[0])

            pids_filtered = []
            for p in pids_unfiltered:
                if known_tracks[p][0][0] == 0 and known_tracks[p][-1][0] == 99:
                    if cat == 9:
                        pids_filtered.append(p)
                else:
                    if cat == 10:
                        pids_filtered.append(p)

            self.__test_pids = pids_filtered

        return self.__test_pids

    def get_test_pids_indices(self, cat='all'):
        """
        Return the indices of the test playlists of the selected category.
        :param cat: ['all', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        :return: cat_indices: list of indices of playlists of selected cat
        """

        pids_all = self.get_test_pids(cat='all')
        pids_cat = self.get_test_pids(cat=cat)

        mask = np.in1d(pids_all, pids_cat, invert=False)
        cat_indices = np.where(mask)[0]

        return list(cat_indices)

    def get_train_pids(self):
        if hasattr(self, '__train_pids'):
            return self.__train_pids
        self.__train_pids =  self.get_train_playlists(_verbose=False).transpose()[0]
        return self.__train_pids

    def get_df_train_interactions(self):
        if self.__verbose: print("[reading TRAIN INTERACTIONS from "+self.__path+self.__train_interactions_file+" ] ", end="")
        df_train = pd.read_csv(filepath_or_buffer=self.__path + self.__train_interactions_file,
                               sep=self.__separator,header=0, usecols=['pid', 'tid', 'pos'],
                               dtype={'pid': self.__dtype_int, 'tid': self.__dtype_int, 'pos': self.__dtype_int})
        if self.__verbose: print('>DF train interactions read')
        return df_train

    def get_track_to_artist_dict(self):
        """
        :return: dictionary: {track: artist}
        """
        tracks_df = self.get_df_tracks()

        keys = list(tracks_df['tid'].as_matrix())
        values = list(tracks_df['arid'].as_matrix())
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary

    def get_artist_id_to_uri_dict(self):
        artist_df = self.get_df_artists()

        keys = list(artist_df['arid'].as_matrix())
        values = list(artist_df['uri'].as_matrix())
        dictionary = dict(zip(keys, values))
        del artist_df

        return dictionary

    def get_track_id_to_uri_dict(self):
        track_df = self.get_df_tracks()

        keys = list(track_df['tid'].as_matrix())
        values = list(track_df['uri'].as_matrix())
        dictionary = dict(zip(keys, values))
        del track_df

        return dictionary

    def get_artist_to_tracks_dict(self):
        """
        :return: dictionary: {artist: [track1, track2, ...]}
        """
        tracks_df = self.get_df_tracks()
        dictionary = dict()

        for index, row in tqdm(tracks_df.iterrows(), desc='Creating dictionary artist to tracks'):

            arid = row['arid']
            tid = row['tid']

            if arid in dictionary.keys():
                dictionary[arid].append(tid)
            else:
                dictionary[arid] = [tid]

        return dictionary

    def get_track_to_album_dict(self):
        """
        :return: dictionary: {track: album}
        """
        tracks_df = self.get_df_tracks()

        keys = list(tracks_df['tid'].as_matrix())
        values = list(tracks_df['alid'].as_matrix())
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary

    def get_album_to_tracks_dict(self):
        """
        :return: dictionary: {album: [track1, track2, ...]}
        """
        tracks_df = self.get_df_tracks()
        dictionary = dict()

        for index, row in tqdm(tracks_df.iterrows(), desc='Creating dictionary album to tracks'):

            alid = row['alid']
            tid = row['tid']

            if alid in dictionary.keys():
                dictionary[alid].append(tid)
            else:
                dictionary[alid] = [tid]

        return dictionary

    def get_pid_to_name_dict(self):
        train_playlists_df = self.get_df_train_playlists()
        test_playlists_df = self.get_df_test_playlists()
        concat_df = pd.concat([train_playlists_df, test_playlists_df])

        keys = list(concat_df['pid'].as_matrix())
        values = list(concat_df['name'].as_matrix())
        dictionary = dict(zip(keys, values))

        return dictionary

    def get_playlists_popularity(self):
        """
        Compute the popularity for each playlist as the mean of popularity for each track.
        :return: playlists_pop: a Numpy array of popularity for each playlist
        """
        urm = self.get_urm(binary=True)
        track_pop = urm.sum(axis=0).A1

        playlists_pop = []

        for idx in tqdm(range(urm.shape[0]), desc='Playlists popularity'):
            start = urm.indptr[idx]
            end = urm.indptr[idx+1]

            tracks = urm.indices[start:end]
            pops = [track_pop[t] for t in tracks]

            if len(pops) == 0:
                playlists_pop.append(0)
            else:
                playlists_pop.append(np.mean(pops))

        return np.array(playlists_pop)

    def get_playlists_popularity_clusters(self, n_clusters):
        pop = self.get_playlists_popularity()

        playlists_pop = np.argsort(pop)
        popularity_sorted = np.sort(pop)

        playlists_per_cluster = int((len(playlists_pop) - 1000) / n_clusters)

        dict_cluster = dict()
        clusters_tresholds = []

        for i in range(n_clusters):
            start = int(1000 + i * playlists_per_cluster)
            end = start + playlists_per_cluster

            dict_cluster[i] = list(playlists_pop[start:end])

            if end >= 1000000:
                clusters_tresholds.append(popularity_sorted[-1])
            else:
                clusters_tresholds.append(popularity_sorted[end])

        return clusters_tresholds, dict_cluster

    def get_eurm_top_pop_playlist(self, eurm_nlp):
        urm = self.get_urm(binary=True)
        pop_tracks = urm.sum(axis=0).A1

        clusters_tresholds, dict_cluster = self.get_playlists_popularity_clusters(n_clusters=1000)

        test_pids = self.get_test_pids(cat=1)

        rows = []
        cols = []
        data = []

        for idx in tqdm(range(len(test_pids)), desc='Eurm pop cat1'):
            start = eurm_nlp.indptr[idx]
            end = eurm_nlp.indptr[idx+1]

            tracks_pop = [pop_tracks[t] for t in eurm_nlp.indices[start:end]]
            playlist_pop = np.mean(tracks_pop)

            i = 0
            cluster = 0
            while playlist_pop <= clusters_tresholds[i]:
                cluster = i
                i += 1

            pop_filtered = urm[dict_cluster[cluster]].sum(axis=0).A1

            top_pop_filtered = np.argsort(pop_filtered)[::-1][:500]

            for t in top_pop_filtered:
                rows.append(idx)
                cols.append(t)
                data.append(pop_filtered[t])

        eurm_top = sp.csr_matrix((data, (rows, cols)), shape=eurm_nlp.shape)

        return eurm_top

    def get_df_eval_interactions(self, ):
        assert self.__offline() , "you must run the datareader for Testing purpose. (OFFLINE)"
        assert 'test' in self.__path
        # if self.__verbose: print("[reading EVAL INTERACTIONS: csv from: "+self.__path+self.__eval_interactions_file+" ]", end="")
        df_eval = pd.read_csv(filepath_or_buffer=self.__path + self.__eval_interactions_file,
                              sep=self.__separator, header=0 )
        # if self.__verbose: print('>DF EVAL interactions read')
        return df_eval

    def get_df_test_interactions(self):
        # if self.__verbose: print("[reading TEST INTERACTIONS from " + self.__path + self.__test_interactions_file + " ] ", end="")
        df_test = pd.read_csv(filepath_or_buffer=self.__path + self.__test_interactions_file,
                          sep=self.__separator, header=0, usecols=['pid', 'tid', 'pos'],
                          dtype={'pid': self.__dtype_int, 'tid': self.__dtype_int, 'pos': self.__dtype_int})
        # if self.__verbose: print('>DF TEST interactions read')
        return df_test

    def get_df_train_playlists(self):
        # if self.__verbose: print("[reading TRAIN PLAYLISTS from " + self.__path + self.__train_playlists_file + " ]", end="")
        df = pd.read_csv(filepath_or_buffer=self.__path + self.__train_playlists_file, sep=self.__separator)
        # if self.__verbose: print('>DF TRAIN interactions read')
        return df

    def get_df_test_playlists(self):
        # if self.__verbose: print("[reading TEST PLAYLISTS from " + self.__path + self.__test_playlist_file + " ]")
        df = pd.read_csv(filepath_or_buffer=self.__path + self.__test_playlist_file, sep=self.__separator)
        return df

    def get_df_test_albums(self):
        # if self.__verbose: print("reading ALBUMS from " + self.__path_original_csv + self.__albums_file + " ]")
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__albums_file, sep=self.__separator)
        return df

    def get_df_tracks(self):
        # if self.__verbose: print("[reading TRACKS from " + self.__path_original_csv + self.__tracks_file + " ]")
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__tracks_file, sep=self.__separator,
                         header=0)
        return df

    def get_df_artists(self):
        # if self.__verbose: print("[reading ARTISTS from " + self.__path_original_csv + self.__artists_file + " ]")
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__artists_file, sep=self.__separator,
                         header=0)
        return df

    def get_df_albums(self):
        # read file
        df = pd.read_csv(filepath_or_buffer=self.__path_original_csv + self.__albums_file, sep=self.__separator,
                         header=0,
                         usecols=['alid', 'album_name'],
                         dtype={'alid': self.__dtype_int, 'album_name': self.__dtype_str})
        return df

    def __save_matrix(self, name, sparse_matrix):
        if not os.path.exists(self.__path + 'matrices/'):
            if self.__verbose: print("[creating data folder in " + self.__path_matrices + " ]")
            os.makedirs(self.__path_matrices)
        if self.__verbose: print("[saving "+name+".npz in "+self.__path_matrices+ " ]")
        sp.save_npz(self.__path_matrices + name+'.npz', sparse_matrix)

    def __load_matrix(self, name):
        return sp.load_npz(self.__path_matrices+name+'.npz').tocsr()

    def __save_dictionary(self, name, file ):
        with open(self.__path_matrices+name+'.pickle', 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_dictionary(self, name ):
        with open(self.__path_matrices+name+'.pickle', 'rb') as handle:
            return pickle.load(handle)

    def __build_name(self, name, format='', to_add=''):
       urm_npz_name = name
       if len(format) > 0:
           urm_npz_name += '_' + format
       if len(to_add) > 0:
           urm_npz_name += '_' + to_add
       return urm_npz_name

    def __online(self):
        return self.__online_or_offline == 'online'

    def __offline(self):
        return self.__online_or_offline == 'offline'

    def online(self):
        return self.__online_or_offline == 'online'

    def offline(self):
        return self.__online_or_offline == 'offline'

    def info(self):
        print('!! URM, playlist contains dupilcate tracks (so cell value != 1), adjust this value in preprocessing !!')

    def get_last_n_songs_urm(self, n=5):

        pos_matrix = self.get_position_matrix(position_type='last')
        for i in tqdm(range(pos_matrix.shape[0])):
            data= pos_matrix.data[ pos_matrix.indptr[i]:pos_matrix.indptr[i+1] ]
            if len(data)>0:
                max_data = np.max( data )
                mask = [data<= max_data-n]
                pos_matrix.data[ pos_matrix.indptr[i]:pos_matrix.indptr[i+1] ][mask]    = 0
        pos_matrix.eliminate_zeros()
        print("rimasti> dovrebbe esser poco meno di 1'000'000 *",n,":",pos_matrix.nnz)
        return  pos_matrix

    def get_popularity_playlists(self):
        urm = self.get_urm()

        tracks_popularity = np.ravel(urm.sum(axis=0))
        playlists_popularity = []

        for row in tqdm(range(urm.shape[0]), desc='Playlists popularity'):
            row_start = urm.indptr[row]
            row_end = urm.indptr[row + 1]

            indices = urm.indices[row_start:row_end]
            if len(indices) > 0:
                mean = np.mean(tracks_popularity[indices])
            else:
                mean = 0
            playlists_popularity.append(mean)

        return np.array(playlists_popularity)

    def get_popularity_tracks(self):
        urm = self.get_urm()
        tracks_popularity = np.ravel(urm.sum(axis=0))

        return tracks_popularity
    
    #################################
    ## for creative track only
    
    def get_icm_refined_feat(self, feat=None, K= 4, load_only=True, mode='all_line'):
        if load_only:
            file_name = mode+'_refine_arid_'+feat+'.npz'
            icm = sp.load_npz( self.__path_enriched_icm + file_name)
        else:
            # TODO
            print('No magic yet... please go generate layered matrix first.')
        return icm.tocsr()

    def get_icm_refined_pid_feat(self, feat=None, K=4, load_only=True, mode='offline'):
        
        if load_only:     
            file_name =  mode+'_refine_pid_'+feat+'.npz'
            # load csr
            icm = sp.load_npz( self.__path_enriched_icm + file_name)
        else:
            # TODO
            print('No magic yet... please go generate layered matrix first.')
        return icm.tocsr()

    ## Creative above
    ##########################

if __name__ == '__main__':

    dr = Datareader(mode='online',   only_load=True)

    dr.get_last_n_songs_urm(n=5)


