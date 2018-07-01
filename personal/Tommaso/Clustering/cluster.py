from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from scipy import sparse
import numpy as np
import utils.pre_processing as pre
from tqdm import tqdm
from utils.definitions import *


class Cluster(object):

    def __init__(self, urm, similarity):
        self.urm = urm
        self.similarity = similarity

        # Adjust matrices
        self.urm.data = np.ones(len(urm.data))
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()
        self.similarity = pre.norm_l2_row(similarity)

        # Compute distance matrix
        self.distance_data = 1. / similarity.data
        self.distance = similarity.copy()
        self.distance.data = self.distance_data
        self.distance = pre.norm_l2_row(self.distance)

    def fit(self, eps=0.05, min_samples=8, verbose=True):

        if verbose:
            print('* Distance matrix statistics *')
            print('min', np.min(self.distance_data))
            print('max', np.max(self.distance_data))
            print('mean', np.mean(self.distance_data))

        # Clustering
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed',
                        metric_params=None, algorithm='auto', leaf_size=5, p=None,
                        n_jobs=-1).fit(self.distance)

        self.labels_unique = list(set(self.dbscan.labels_))
        self.labels_unique.sort()

        if verbose:
            print('items:', len(self.dbscan.labels_))
            print('labels:', len(self.labels_unique))

    def create_dict(self, verbose=True, save=False):

        # Create an empty dict for each unique label
        values = []
        for i in range(len(self.labels_unique)):
            values.append(set())

        self.dictionary = dict(zip(self.labels_unique, values))

        # Create dict
        for track in tqdm(range(len(self.dbscan.labels_)), desc='Create cluster'):
            label = self.dbscan.labels_[track]
            self.dictionary[label].add(track)

        # Cluster statistics
        if verbose:
            lengths = []
            for k in self.dictionary:
                if k != -1:
                    lengths.append(len(self.dictionary[k]))
            print('* Cluster statistics *')
            print('min:', np.min(lengths))
            print('max:', np.max(lengths))
            print('mean:', np.mean(lengths))
            print('noise [-1]:', len(self.dictionary[-1]))
            print('% noise:', len(self.dictionary[-1]) / len(self.dbscan.labels_))

        if save:
            save_obj(self.dictionary, 'cluster_dict', '')
            save_obj(self.dbscan.labels_, 'cluster_labels', '')

        return self.dbscan.labels_, self.dictionary
