from utils.evaluator import Evaluator
from utils.post_processing import *
from utils.pre_processing import *
from utils.submitter import Submitter
from personal.MaurizioFramework.MatrixFactorization.PureSVD import PureSVDRecommender
from utils.datareader import Datareader
from scipy import sparse


def compute_SVD(dr, n_factors, top_k, save_eurm):
    test_pids = dr.get_test_pids()

    # Mode
    if dr.offline():
        mode = 'offline'
    else:
        mode = 'online'

    # URM
    urm = dr.get_urm()
    urm = bm25_row(urm)

    # Train model
    svd_rec = PureSVDRecommender(urm)
    svd_rec.fit(n_factors)

    # Compute predictions
    print('Computing eurm...')
    rows = []
    cols = []
    data = []

    # If online, do not use challenge set
    if mode == 'offline':
        predictions = svd_rec.compute_score_SVD(user_id=test_pids)
    else:
        test_users_sparse = urm[test_pids]
        predictions = svd_rec.compute_score_cold_users_SVD(test_users_sparse=test_users_sparse)

    for i in tqdm(range(len(test_pids)), desc='SVD eurm'):
        relevant_items_partition = (-predictions[i]).argpartition(top_k - 1)[0:top_k]
        relevant_items_partition_sorting = np.argsort(-predictions[i][relevant_items_partition])
        top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

        # Incrementally build sparse matrix, do not add zeros
        notZerosMask = predictions[i][top_k_idx] != 0.0
        numNotZeros = np.sum(notZerosMask)

        data.extend(predictions[i][top_k_idx][notZerosMask])
        cols.extend(top_k_idx[notZerosMask])
        rows.extend(np.ones(numNotZeros) * i)

    eurm = sparse.csr_matrix((data, (rows, cols)), shape=(len(test_pids), urm.shape[1]))

    if save_eurm:
        print('Saving eurm...')
        sparse.save_npz('eurm_svd_bm25_' + str(n_factors) + '_' + mode + '.npz', eurm)

    return eurm


if __name__ == '__main__':

    n_factors = 100
    top_k = 750
    mode = 'online'

    if mode == 'offline':
        # Initialization
        dr = Datareader(mode='offline', only_load=True, verbose=False)
        ev = Evaluator(dr)

        # Prediction
        eurm = compute_SVD(dr, n_factors, top_k, save_eurm=True)

        # Evaluation
        print('N_FACTORS =', n_factors)
        ev.evaluate(eurm_to_recommendation_list(eurm, datareader=dr), name='svd_' + str(n_factors))

    elif mode == 'online':
        # Initialization
        dr = Datareader(mode='online', only_load=True, verbose=False)
        sb = Submitter(dr)

        # Prediction
        eurm = compute_SVD(dr, n_factors, top_k, save_eurm=True)

        # Submission
        sb.submit(eurm_to_recommendation_list_submission(eurm, datareader=dr), name='svd_' + str(n_factors))

    else:
        print('Wrong mode!')



