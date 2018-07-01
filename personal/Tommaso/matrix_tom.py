from fast_import import *
import sys

arg = sys.argv[1:]
#arg = ['offline']
mode = arg[0]
save = True
filename = 'similarity_tom_'+mode+'.npz'

#common part
dr = Datareader(mode=mode, only_load=True, verbose=False)

urm = sp.csr_matrix(dr.get_urm(),dtype=np.float)
rec = CF_IB_BM25_strange(urm=urm, binary=True, datareader=dr, mode=mode, verbose=True, verbose_evaluation= False)
rec.model(alpha=1, beta=0, k=150, shrink=0, threshold=0)
sim = rec.s

if save:
    sp.save_npz(filename ,sim)
