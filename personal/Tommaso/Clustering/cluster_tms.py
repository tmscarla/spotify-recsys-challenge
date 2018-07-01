from scipy import sparse
from utils.definitions import ROOT_DIR
import numpy as np
from tqdm import tqdm
import itertools

urm = sparse.load_npz(ROOT_DIR + '/data/test1/matrices/urm.npz')
urm.data = np.ones(len(urm.data))
print('URM Loaded!')

sim = sparse.load_npz(ROOT_DIR + '/data/sim.npz')
sim.setdiag(0)
sim.eliminate_zeros()
print('SIM Loaded!')

mini_cluster = []

for i in tqdm(range(sim.shape[0] - 1)):
    row_start = sim.indptr[i]
    row_end = sim.indptr[i + 1]

    row_data = sim.data[row_start:row_end]
    row_columns = sim.indices[row_start:row_end]

    if len(row_data) > 0:
        top = np.argsort(row_data)[::-1][0]
        mini_cluster.append((i, row_columns[top]))

print(len(mini_cluster))
print(mini_cluster)

a = list(set(itertools.permutations(mini_cluster, 2)))

print(len(a))


# for i in tqdm(range(urm.shape[0] - 1)):
#     first = urm[i, :].toarray().astype(np.int)[0]
#     second = urm[i+1, :].toarray().astype(np.int)[0]
#
#     common = np.bitwise_and(first, second)
#     print(len(common.nonzero()[0]))
