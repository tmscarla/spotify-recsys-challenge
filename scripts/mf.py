from personal.MaurizioFramework.MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
# from utils.datareader import Datareader
from recommenders.similarity.similarity import *
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
import lightfm.datasets as ld
import sys

# sys.stdout = open("mf_prova.txt", "w")

data = ld.fetch_movielens(min_rating=1.0)
train =sp.csr_matrix( data["train"])
test = sp.csr_matrix(data["test"])
train.data = np.ones(len(train.data))
test.data = np.ones(len(test.data))
print(train.shape, train.nnz)
print(test.shape, test.nnz)

print(len(train.data), np.sum(train.data))
print(len(test.data),np.sum(test.data))

print(train[100])
print(test[100])

epochs = 500
factors = 42
learning = 0.001

gc.collect()

print("start")

rec = MF_BPR_Cython(train)
rec.fit(epochs=epochs, URM_test=test, filterTopPop=False, filterCustomItems=np.array([], dtype=np.int),
        minRatingsPerUser=0, evaluating_at=5,
        batch_size=1000, validate_every_N_epochs=4, start_validation_after_N_epochs=0,
        num_factors=factors, positive_threshold=0,
        learning_rate=learning, sgd_mode='adagrad', user_reg=0.1,
        positive_reg=0.1, negative_reg=0.1)


exit()


# Test case: {'learn_rate': 0.1, 'num_factors': 42, 'batch_size': 1, 'epoch': 0}
# Results {'AUC': 0.020943796394485684, 'precision': 0.008695652173913047, 'recall': 0.0043478260869565235, 'map': 0.003916578296217746, 'NDCG': 0.005593740129458184, 'MRR': 0.01958289148108872}
# Test case: {'learn_rate': 0.1, 'num_factors': 42, 'batch_size': 1, 'epoch': 60}
# Results {'AUC': 0.0581477553905974, 'precision': 0.032661717921526966, 'recall': 0.016330858960763483, 'map': 0.012877341816896417, 'NDCG': 0.018894727961066417, 'MRR': 0.057847295864263064}
# Test case: {'learn_rate': 0.1, 'num_factors': 42, 'batch_size': 1, 'epoch': 112}
# Results {'AUC': 0.06574761399787911, 'precision': 0.030116648992576822, 'recall': 0.015058324496288411, 'map': 0.014630611523506523, 'NDCG': 0.01947096489519987, 'MRR': 0.06378579003181344}
# Test case: {'learn_rate': 0.1, 'num_factors': 42, 'batch_size': 1, 'epoch': 272}
# Results {'AUC': 0.10701661364439732, 'precision': 0.04050901378579003, 'recall': 0.020254506892895013, 'map': 0.02286320254506891, 'NDCG': 0.028996752806142178, 'MRR': 0.10311063980205022}
# Test case: {'learn_rate': 0.1, 'num_factors': 42, 'batch_size': 1, 'epoch': 440}
# Results {'AUC': 0.07476139978791092, 'precision': 0.03541887592788965, 'recall': 0.017709437963944825, 'map': 0.014743725698126521, 'NDCG': 0.021345722847563843, 'MRR': 0.0682396606574762}
# Results {'AUC': 0.09455638034641216, 'precision': 0.03860021208907737, 'recall': 0.019300106044538686, 'map': 0.018480028278543635, 'NDCG': 0.025027912382672473, 'MRR': 0.08354542241074588}


















if __name__ == '__main__asd':
    train_format = ''
    dr = Datareader(mode="offline",train_format="50k", only_load=True)
    urm,a,b = dr.get_urm_shrinked()
    print(urm.shape)
    urm.data = np.ones(len(urm.data))

    urm2 = urm.tocoo().copy()

    msk = np.random.rand(len(urm2.data) ,) < 0.81

    train = sps.coo_matrix( (urm2.data[msk], (urm2.row[msk], urm2.col[msk])),shape=urm.shape,  dtype=np.int, copy=False)
    test = sps.coo_matrix( (urm2.data[~msk], (urm2.row[~msk], urm2.col[~msk])), shape=urm.shape,  dtype=np.int, copy=False)

    train = train.tocsr()
    test = test.tocsr()

    epochs=1000
    factors=200
    learning =0.005

    gc.collect()

    print("start")

    rec = MF_BPR_Cython(train)
    rec.fit(epochs=epochs, URM_test=test, filterTopPop = False, filterCustomItems = np.array([], dtype=np.int),
            minRatingsPerUser=5, evaluating_at=500,
            batch_size = 1000, validate_every_N_epochs = 100, start_validation_after_N_epochs = 0,
            num_factors=factors,  positive_threshold=0,
            learning_rate = learning, sgd_mode='adagrad', user_reg = 0.0,
            positive_reg = 0.0, negative_reg = 0.0)

    pids = dr.get_test_pids()
    W = rec.W[pids]


    #################################### RECOMMENDATIONS ##############################
    at=500
    rows = np.zeros(len(pids) * at)
    cols = np.zeros(len(pids) * at)
    data = np.zeros(len(pids) * at)
    numCells = 0

    tot = list()
    for user in tqdm(pids):

        scores_array = np.dot(rec.W[user], rec.H.T)

        relevant_items_partition = (-scores_array).argpartition(at)[0:at]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        rows[numCells:numCells+at] = np.ones(at) * user
        cols[numCells:numCells+at] = ranking[:at]
        data[numCells:numCells+at] = scores_array[ranking[:at]]

        numCells += at

    R = sps.csr_matrix((data, (rows, cols)))

    sp.save_npz("mf_WEEKEND_"+str(factors)+"fact_"+str(epochs)+"ep_"+str(learning)+"le_"+
                train_format+".npz", R)


