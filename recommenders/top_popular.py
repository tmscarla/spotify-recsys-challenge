from tqdm import tqdm
import scipy.sparse as sps
from recommenders.recommender import Recommender
import numpy as np

class Top_pop(Recommender):
    def __init__(self):
        Recommender.__init__(self)
        self.icm = None

    def fit(self, urm, pid):
        self.urm = urm
        self.pid = pid

    def compute_model(self):
        pass

    def compute_rating(self, top_k=500):


        self.urm.data = np.ones(len(self.urm.data))
        top = sps.csc_matrix(self.urm).sum(axis=0).A1
        ind = top.argsort()[-top_k:][::-1]
        self.eurm = sps.lil_matrix((len(self.pid), self.urm.shape[1]))
        print(self.eurm.shape)
        for i in tqdm(range(len(self.pid))):
            self.eurm[i, [ind]] = top[ind]

        #TODO: remove after update
        import warnings
        warnings.warn('This function still use the old version of the remove seed, it should be replaced soon by the one in post_processing class')


        self.urm = sps.csr_matrix(self.urm[self.pid])
        tmp = self.urm.tocoo()
        row = tmp.row
        col = tmp.col
        min = self.eurm.tocoo().min()
        self.eurm = sps.lil_matrix(self.eurm)
        self.eurm[row, col] = -1
        self.eurm = sps.csr_matrix(self.eurm)

        return self.eurm





if __name__ == '__main__':
    from utils.datareader import Datareader
    dr = Datareader(verbose=False, mode="offline", only_load="False")

    rec = Top_pop()
    rec.fit(dr.get_urm(), dr.get_test_playlists().transpose()[0])
    eurm = rec.compute_rating().tocsr()
    sps.save_npz("top_pop online.npz", eurm.tocsr())
    exit()
    import utils.evaluator as ev
    from utils.post_processing import eurm_to_recommendation_list
    eva = ev.Evaluator(dr)

    eva.evaluate( eurm_to_recommendation_list(eurm),"cacca TOPTOP")