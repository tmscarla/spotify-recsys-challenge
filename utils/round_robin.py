import numpy as np
from tqdm import tqdm

class RoundRobin(object):

    def __init__(self, rec_lists, weights=None):
        """
        Initialize the round robin ensembler
        :param rec_lists: a list of rec_list
        :param weights: a list of weights for each recommender
        """
        self.rec_lists = rec_lists

        # By default weight equally each recommender
        if weights is None:
            self.weights = list(np.ones(len(rec_lists)))
        else:
            self.weights = weights

        try:
            if len(self.rec_lists) != len(self.weights):
                raise ValueError('Error: predictions and weights length must match')
        except ValueError as error:
            print('Caught this error: ' + repr(error))

    def __init_indices(self, playlist_index):

        # Gather predictions for the specified playlist index
        self.predictions = []
        for r in self.rec_lists:
            self.predictions.append(r[playlist_index])

        # Initialize indices
        self.i = 0
        self.last_indices = np.zeros(len(self.predictions)).astype(np.int)
        self.top_K_indices = []

    def rr_std(self, playlist_index, K):
        """
        RR in standard mode: pick the best track from each recommendation
        :param playlist_index: the index of the playlist for each rec_list
        :param K: number of tracks to be returned
        :return: top_K_indices
        """

        # Initialization
        self.__init_indices(playlist_index)

        while self.i < K:
            indices = np.zeros(len(self.predictions)).astype(np.int)

            for p_i in range(len(self.predictions)):

                while indices[p_i] != self.weights[p_i]:

                    p_last_index = self.predictions[p_i][self.last_indices[p_i]]

                    if p_last_index not in self.top_K_indices:
                        self.top_K_indices.append(p_last_index)
                        indices[p_i] += 1
                        self.i += 1
                    self.last_indices[p_i] += 1

                    # Jump to the next recommender if no suitable track is found
                    if self.last_indices[p_i] == 500:
                        indices[p_i] = self.weights[p_i]

        return self.top_K_indices

    def rr_jmp(self, playlist_index, K):
        """
        RR in jump mode: if the tracks is already selected, jump to the next recommendation
        :param playlist_index: the index of the playlist for each rec_list
        :param K: number of tracks to be returned
        :return: top_K_indices
        """

        # Initialization
        self.__init_indices(playlist_index)

        while self.i < K:
            for p_i in range(len(self.predictions)):

                for j in range(0, self.weights[p_i]):
                    p_last_index = self.predictions[p_i][self.last_indices[p_i]]

                    if p_last_index not in self.top_K_indices:
                        self.top_K_indices.append(p_last_index)
                        self.i += 1
                    self.last_indices[p_i] += 1

        return self.top_K_indices

    def rr_mono(self, playlist_index, K):
        """
        RR in mono mode: pick exactly one track for each recommendation
        :param playlist_index: the index of the playlist for each rec_list
        :param K: number of tracks to be returned
        :return: top_K_indices
        """

        # Initialization
        self.__init_indices(playlist_index)

        while self.i < K:
            for p_i in range(len(self.predictions)):
                p_last_index = self.predictions[p_i][self.last_indices[p_i]]

                if p_last_index not in self.top_K_indices:
                    self.top_K_indices.append(p_last_index)
                    self.i += 1
                self.last_indices[p_i] += 1

        return self.top_K_indices

    def rr_avg(self, playlist_index, rec_index, cut_off, K):
        """
        RR in average mode: for each track, compute the average of rankings and pick the top K
        :param playlist_index: the index of the playlist for each rec_list
        :param rec_index: index of the rec_list chosen to iterate the tracks
        :param cut_off: number of tracks to be selected from the chosen rec_list
        :param K: number of tracks to be returned
        :return: top_K_indices
        """

        # Initialize
        self.__init_indices(playlist_index)
        avg_list = []

        for i in self.predictions[rec_index][:cut_off]:
            sum = 0

            for p_i in range(len(self.predictions)):
                track_pos = np.where(self.predictions[p_i] == i)[0]

                if len(track_pos) > 0:
                    sum += track_pos[0]
                else:
                    sum += 500

            avg = sum / len(self.predictions)
            avg_list.append(avg)

        avg_list_argsorted = np.argsort(avg_list)

        for a in avg_list_argsorted[:K]:
            self.top_K_indices.append(self.predictions[rec_index][a])

        return self.top_K_indices
