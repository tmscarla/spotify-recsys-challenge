from fast_import import *
import sys
from utils.datareader import ROOT_DIR


def generate_similarity(mode):
    """
    Generate a similarity matrix (tracks x tracks) and save it on /data
    :param mode: 'offline' or 'online'
    """
    save = True
    filename = 'similarity_tracks_'+mode+'.npz'

    dr = Datareader(mode=mode, only_load=True, verbose=False)

    urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
    rec = CF_IB_BM25_strange(urm=urm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)
    rec.model(alpha=1, beta=0, k=150, shrink=0, threshold=0)
    sim = rec.s

    if save:
        sp.save_npz(ROOT_DIR + '/data/' + filename, sim)


if __name__ == '__main__':
    arg = sys.argv[1:]
    mode = arg[0]
    generate_similarity(mode)
