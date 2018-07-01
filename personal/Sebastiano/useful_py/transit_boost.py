import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
from utils.pre_processing import norm_max_row


def transitivity_boost(sim):
    sim_col = sps.csc_matrix(sim)
    print("Similarity shape: " + str(sim.shape))

    for row in tqdm(range(sim.shape[0])):
        data_row = sim.data[sim.indptr[row]:sim.indptr[row + 1]]
        t_max = np.argwhere(data_row < 0.05).ravel()
        row_indices = sim.indices[sim.indptr[row]:sim.indptr[row + 1]]

        for ind in t_max:
            col = row_indices[ind]
            col_indices = sim_col.indices[sim_col.indptr[col]:sim_col.indptr[col + 1]]
            data_col = sim_col.data[sim_col.indptr[col]:sim_col.indptr[col + 1]]

            data_com_col = data_col[np.where(np.isin(col_indices, row_indices))[0]]
            data_com_row = data_row[np.where(np.isin(row_indices, col_indices))[0]]

            data_row[ind] = np.max(data_com_row + data_com_col) / 2  # TODO: fai la media invece del max
        sim.data[sim.indptr[row]:sim.indptr[row + 1]] = data_row
    return sim

if __name__ == '__main__':

    print("[ Loading Similarity ]")
    sim = sps.csr_matrix(norm_max_row(sps.load_npz("../../scripts/rp3beta_similarity_online.npz").tocsr()))
    boosted = transitivity_boost(sim)
    sps.save_npz("../../scripts/boosted_rp3beta_similarity.npz", boosted)


    # for row in tqdm(range(sim.shape[0])):
    #     i = sim[row].toarray().ravel()
    #     data = sim.data[sim.indptr[row]:sim.indptr[row+1]]
    #     indices = sim.indices[sim.indptr[row]:sim.indptr[row+1]]
    #     for col in (range(len(data))):
    #         if data[col] < t_max :
    #             j = sim_col[:, indices[col]].toarray().ravel()
    #             data[col] = np.max((i+j)/2)
    #     sim.data[sim.indptr[row]:sim.indptr[row + 1]] = data
    # sps.save_npz("../../scripts/boosted_rp3beta_similarity.npz", sim)


