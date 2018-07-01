import sys
from scipy import sparse
import numpy as np
import utils.pre_processing as pre
from utils.definitions import *
from utils.datareader import Datareader
from utils.evaluator import Evaluator
from utils.pre_processing import *
from utils.post_processing import *

dr = Datareader(mode='offline', only_load=True, verbose=False)
ev = Evaluator(dr)

urm = dr.get_urm(binary=True)
pos_matrix = dr.get_position_matrix(position_type='last')

rows = []
cols = []
data = []

for p in tqdm(range(pos_matrix.shape[0])):
    start = pos_matrix.indptr[p]
    end = pos_matrix.indptr[p+1]

    tracks = pos_matrix.indices[start:end]
    positions = pos_matrix.indices[start:end]

    for idx in range(len(tracks)):
        if positions[idx] <= 250:
            rows.append(p)
            cols.append((tracks[idx] * positions[idx]) + tracks[idx])
            data.append(1)

urm_pos_ext = sparse.csr_matrix((data, (rows, cols)), shape=(pos_matrix.shape[0], 250 * pos_matrix.shape[1]))

new_urm = sparse.hstack((urm, urm_pos_ext))
sparse.save_npz('new_urm.npz', new_urm)
