import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp


import gc

TESTNUM='1'
PLAYLISTS_ELIMINATED = 10000

class TestTestSet:

    train_interactions= pd.read_csv(filepath_or_buffer="../data/test"+TESTNUM+"/train_interactions.csv", delimiter='\t')
    test_interactions= pd.read_csv(filepath_or_buffer="../data/test"+TESTNUM+"/test_interactions.csv", delimiter='\t')

    def test_test_playlists(self):

        # check on the challange playlists
        df = pd.read_csv("../data/test"+TESTNUM+"/test_playlists.csv",delimiter ='\t')
        assert len( df[(df['num_samples']==0)])==1000 , "err1"
        assert len( df[(df['num_samples']==1)])==1000 , "err2"
        assert len( df[(df['num_samples']==5)])==2000 , "err3"
        assert len( df[(df['num_samples']==10)])==2000 , "err4"
        assert len( df[(df['num_samples']==25)])==2000 , "err5"
        assert len( df[(df['num_samples']==100)])==2000 , "err6"
        assert len(df)==10000 ,"err all"
        del(df)

    def test_train_playlists(self):

        df = pd.read_csv("../data/test"+TESTNUM+"/train_playlists.csv",delimiter ='\t')
        assert len(df)==(1000000 - PLAYLISTS_ELIMINATED) , "should not contain the test playlists"+str(len(df))

    def test_test_interactions(self):
        print("test interactions ")

        playlists = self.test_interactions['pid'].values
        tracks = self.test_interactions['tid'].values
        print(tracks.size)
        n_interactions = tracks.size
        n_playlists = playlists.max() + 1  # index starts from 0
        print(n_playlists, "n2")
        # assert n_playlists == 1
        n_tracks = tracks.max() + 1  # index starts from 0
        print(n_tracks, "n2")
        # assert n_tracks == 1
        print(n_interactions, end=' ')
        matrix = sp.csr_matrix((np.ones(n_interactions), (playlists, tracks)), shape=(1000000, 2262292),
                            dtype=np.int32)
        print(len(matrix.data),np.sum(matrix.data),np.sum(matrix.data)-len(matrix.data))
        print("test interactions\t", matrix.nnz, matrix.shape)

        assert n_interactions == 281000, "origin interactions"
        assert np.sum(matrix.data) == 281000, "all interactions are in the matrix"

        assert len(matrix.data)== 278816, "uniques"
        assert np.sum(matrix.data) - len(matrix.data) == 2184,  "duplicates"
        assert matrix.shape == (1000000, 2262292)
        del (matrix)

    def test_train_interactions(self):

        print("train intractions")
        playlists = self.train_interactions['pid'].values
        tracks = self.train_interactions['tid'].values

        print(tracks.size, end=' ')
        n_interactions = tracks.size
        n_playlists = playlists.max() + 1  # index starts from 0
        n_tracks = tracks.max() + 1  # index starts from 0
        print(n_interactions, end=' ')
        matrix = sp.csr_matrix((np.ones(n_interactions), (playlists, tracks)), shape=(n_playlists, n_tracks),
                           dtype=np.int32)
        print(len(matrix.data), np.sum(matrix.data), np.sum(matrix.data) - len(matrix.data))
        print("interactions normali\t", matrix.nnz, matrix.shape)

        assert n_interactions == 65361060 , "origin interactions"
        assert np.sum(matrix.data) == 65361060, "all interactions are in the matrix"

        assert len(matrix.data)== 64494175,  "uniques"

        assert np.sum(matrix.data) - len(matrix.data) == 866885 , "duplicates"
        assert matrix.shape == (1000000, 2262292)
        del (matrix)

    def test_eval_interactions(self):
        print("eval")

        ev = pd.read_csv(filepath_or_buffer="../data/test" + TESTNUM + "/eval_interactions.csv", delimiter='\t')
        playlists = ev['pid'].values
        tracks = ev['tid'].values
        n_interactions = tracks.size

        max_playlist = playlists.max() + 1  # index starts from 0
        assert max_playlist == 999837
        max_track = tracks.max() + 1  # index starts from 0
        assert max_track == 2261344

        print(n_interactions, end=' ')
        matrix = sp.csr_matrix((np.ones(n_interactions), (playlists, tracks)), shape=(1000000, 2262292),
                            dtype=np.int32,)
        print(len(matrix.data),np.sum(matrix.data),np.sum(matrix.data)-len(matrix.data))
        print("eval interactions\t", matrix.nnz, matrix.shape)

        assert n_interactions == 704368 , "origin interactions"
        assert np.sum(matrix.data) == 704368, "all interactions are in the matrix"
        assert len(matrix.data)== 696048,  "uniques"
        assert np.sum(matrix.data) - len(matrix.data) == 8320 , "duplicates"
        assert matrix.shape == (1000000, 2262292)

        # check to have eliminated the right ones
        test_pl = pd.read_csv("../data/test" + TESTNUM + "/test_playlists.csv", delimiter='\t')
        for index, row in tqdm(test_pl.iterrows()):
            assert (row['num_tracks'] - row['num_samples']) == len(ev[ev['pid'] == row['pid']]),\
                "oh boi - check to have eliminated the right ones"

        test_pl['tolte'] = test_pl['num_tracks'] - test_pl['num_samples']

        print("eliminated songs:"+str(np.sum(test_pl['tolte'].values)) + "  " + str(len(ev)))
        assert np.sum(test_pl['tolte'].values) == 704368
        assert len(ev) == 704368
            #"the number of eliminated is equal of evaluation itr ?" + str(np.sum(ev['tolte'].values)) + "  " + str(len(ev))

        # the number of songs taken from interaction is exactly the difference between the two interaction csvs
        print(np.sum(test_pl['num_tracks'].values))
        assert np.sum(test_pl['num_tracks'].values) == 985368, "the number of songs taken from interaction is exactly the difference between the two interaction csvs."
        # hardcoded becouse the files are 1GB each   985368 = 704368 +281000

        print(np.sum(test_pl['num_samples'].values))
