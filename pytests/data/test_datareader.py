from utils.datareader import *
import gc
import pytest

testnum="2"

PLAYLISTS =1000000
PLAYLISTS_FULL = 1049361
PLAYLISTS_TEST = 10000
INTERACTIONS_TRAIN_ORIGINAL = 66346428
INTERACTIONS_TRAIN_TEST1 = 65361060

INTERACTIONS_TEST = 281000
UNIQUE_TRACKS = 2262292
UNIQUE_ALBUMS = 734684
UNIQUE_ARTISTS = 295860
UNIQUE_TITLES = 92944
DESCRIPTIONS = 18760

verbose = True


@pytest.mark.parametrize('execution_number', range(2))
def run_multiple_times(execution_number):
    assert True


class TestDatarReaderONLINE:
    gc.collect()
    dr = Datareader(mode='online', only_load=True, verbose=verbose)

    def test_urm_normal(self):
        urm = self.dr.get_urm(save_on_disk=True)
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists

        print(urm.shape, urm.nnz, np.sum(urm.data), nonempty_playlists)
        assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_ORIGINAL
        assert urm.shape == (PLAYLISTS_FULL, UNIQUE_TRACKS)
        assert nonempty_playlists == PLAYLISTS + PLAYLISTS_TEST - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")


class TestDatarReaderOFFLINE:
    print("############################## TEST OFFLINE full ########################")
    gc.collect()
    dr = Datareader(mode='offline', test_num=testnum, train_format='', only_load=True, verbose=verbose)

    def test_urm_normal(self):
        urm = self.dr.get_urm(save_on_disk=True)
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists
        print(urm.shape, urm.nnz, np.sum(urm.data), nonempty_playlists)
        print( INTERACTIONS_TRAIN_TEST1, INTERACTIONS_TEST, INTERACTIONS_TRAIN_TEST1+INTERACTIONS_TEST, np.sum(urm.data))
        assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_TEST1
        assert urm.shape == (PLAYLISTS, UNIQUE_TRACKS)
        assert nonempty_playlists == PLAYLISTS - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")
        gc.collect()

    def test_icm(self):
        icm = self.dr.get_icm(arid=True, alid=True)
        nonempty_playlists = np.count_nonzero(np.diff(icm.indptr))  # num of nonempty playlists

        print(icm.shape, icm.nnz, np.sum(icm.data), nonempty_playlists)
        assert np.sum(icm.data) == UNIQUE_TRACKS *2
        assert icm.shape == (UNIQUE_TRACKS, UNIQUE_ARTISTS+UNIQUE_ALBUMS)
        assert nonempty_playlists == UNIQUE_TRACKS
        gc.collect()


    def test_eval(self):
        eval = self.dr.get_df_eval_interactions()
        assert 281000 == len(eval)
        pids = self.dr.get_test_pids()
        assert 10000 == len(pids)

    def test_tracks_npy(self):
        tracks = self.dr.get_tracks(name=True, duration=True)
        print(tracks.shape)
        assert tracks.shape == (UNIQUE_TRACKS,3)
        gc.collect()

    def test_tracks_df(self):
        tracks_df = self.dr.get_df_tracks()
        gc.collect()
        #TODO

    def test_train_playlists_npy(self):
        gc.collect()
        pl = self.dr.get_train_playlists(name=True, description=True, num_followers=True, num_tracks=True,
                                         collaborative=False,duration_ms=True,
                                         num_albums=True, num_artists=True, modified_at=True, num_edits=True ,albums_rate=True, artists_rate=True)
        # Indexorder: 0 - pid, 1 - name, 2 - description, 3 - num_followers, 4 - num_tracks,
        # 5 - albums, 6 - num_artists, 7 - modified_at, 8 - num_edits, 9 - duration_ms  10-albums_rate, 11-artists_rate

        for playlist in pl:
            assert playlist[3]>=1 , str(playlist)+" check followers"
            assert playlist[4]>=5 , str(playlist)+" check num tracks"
            assert playlist[10]<=1 and playlist[10]>0, str(playlist)+" check album rate"
            assert playlist[11]<=1 and playlist[11]>0, str(playlist)+" check artist rate"

        assert pl.shape == (PLAYLISTS-PLAYLISTS_TEST , 12)



    def test_test_playlists_npy(self):
        pl = self.dr.get_test_playlists(name=True, num_holdouts=True,
                                          num_samples=True,	num_tracks=True)
        print(pl.shape, PLAYLISTS_TEST)
        #0-pid, 1-name, 2-num_tracks, 3-num_holdouts, 4-num_samples, 5-num_tracks
        assert pl.shape == (PLAYLISTS_TEST, 6)
        for playlist in pl:
            assert playlist[2] == playlist[3] + playlist[4]
        gc.collect()

    def test_artists(self):
        gc.collect()
        #TODO

    def test_albums(self):
        gc.collect()
        #TODO



class TestDatarReaderOFFLINE50k:
    print("############################## TEST OFFLINE 50k ########################")
    gc.collect()
    dr = Datareader(mode='offline', test_num=testnum, train_format='50k', only_load=True, verbose=verbose)

    def test_urm_normal_50k(self):

        urm = self.dr.get_urm(save_on_disk=True)
        print(urm.shape, urm.nnz, np.sum(urm.data))
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists
        # assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_50k
        assert urm.shape == (PLAYLISTS, UNIQUE_TRACKS)
        assert nonempty_playlists == 50000 + 10000 - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")

class TestDatarReaderOFFLINE100k:
    print("############################## TEST OFFLINE 100k ########################")
    gc.collect()
    dr = Datareader(mode='offline', test_num=testnum, train_format='100k', only_load=True, verbose=verbose)

    def test_urm_normal_100k(self):

        urm = self.dr.get_urm(save_on_disk=True)
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists

        print(urm.shape, urm.nnz, np.sum(urm.data), nonempty_playlists)
        # assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_100k
        print(urm.shape)
        print(PLAYLISTS, UNIQUE_TRACKS)
        assert urm.shape == (PLAYLISTS, UNIQUE_TRACKS)
        assert nonempty_playlists ==  100000 + 10000 - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")

class TestDatarReaderOFFLINE200k:
    print("############################## TEST OFFLINE 200k ########################")
    gc.collect()
    dr = Datareader(mode='offline', test_num=testnum, train_format='200k', only_load=True, verbose=verbose)

    def test_urm_normal_200k(self):

        urm = self.dr.get_urm(save_on_disk=True)
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists

        print(urm.shape, urm.nnz, np.sum(urm.data), nonempty_playlists)
        print(PLAYLISTS, UNIQUE_TRACKS)
        # assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_200k
        assert urm.shape == (PLAYLISTS, UNIQUE_TRACKS)
        assert nonempty_playlists ==  200000 + 10000 - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")

class TestDatarReaderOFFLINE400k:
    print("############################## TEST OFFLINE 400k ########################")
    gc.collect()
    dr = Datareader(mode='offline', test_num=testnum, train_format='400k', only_load=True, verbose=verbose)

    def test_urm_normal_400k(self):

        urm = self.dr.get_urm(save_on_disk=True)
        nonempty_playlists = np.count_nonzero(np.diff(urm.indptr))  # num of nonempty playlists

        print(urm.shape, urm.nnz, np.sum(urm.data), nonempty_playlists)
        # assert np.sum(urm.data) == INTERACTIONS_TEST + INTERACTIONS_TRAIN_400k
        assert urm.shape == (PLAYLISTS, UNIQUE_TRACKS)
        assert nonempty_playlists ==  400000 + 10000 - 1000
        print(">>>>>>>>>>>>>>>>>>>TEST URM NORMAL DONE>>>>>>>>>>>>>>>>> ")