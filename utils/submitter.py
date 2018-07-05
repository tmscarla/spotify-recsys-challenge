import pandas as pd
import time
from tqdm import tqdm
import csv
import datetime
import numpy as np
from utils.verify_submission import verify_submission
from utils.definitions import ROOT_DIR
from utils.datareader import Datareader
import os
import sys


class Submitter(object):

    def __init__(self, datareader):
        """
        Initialize the submitter with Dataframes from csv files.
        :param datareader: a Datareader object properly initialized with the full dataset.
        """
        self.tracks = datareader.get_df_tracks()
        self.challenge_playlists = datareader.get_df_test_playlists()['pid'].as_matrix()

        # Create dict track_id - track_uri
        keys = list(self.tracks['tid'].as_matrix())
        values = list(self.tracks['track_uri'].as_matrix())
        self.dictionary = dict(zip(keys, values))

    def convert(self, recommendation_list):
        """
        Convert a numpy matrix of recommendations into a Dataframe.
        :param recommendation_list: a numpy array of recommendations with shape=(10.000,500)
        :return: submission_df: a Dataframe suitable to be saved as csv
        """
        submission_list = []

        for i in tqdm(range(len(recommendation_list)), desc='Converting pids in uris'):
            p = self.challenge_playlists[i]
            rec = recommendation_list[i]

            tracks_uri = [self.dictionary[t] for t in rec]
            submission_list.append([str(p)] + tracks_uri)

        submission_df = pd.DataFrame(submission_list)

        return submission_df

    def submit(self, recommendation_list, name, track='main', verify=True, gzipped=False):
        """
        Take a list of recommendations and make a csv file suitable for submission.
        :param recommendation_list: a numpy array of recommendations with shape=(10.000,500)
        :param name: name of the csv file to be saved
        :param track: ['main', 'creative']
        :param verify: verify the submission before saving it
        :param gzipped: gzip the csv file
        """

        # Create submission dataframe
        submission_df = self.convert(recommendation_list)

        test_header = 'team_info,' + track + ',Creamy Fireflies,creamy.fireflies@gmail.com'
        file_name = name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + '.csv'

        if gzipped:
            file_name = file_name + '.gz'

        # Open the file to write the header
        f = open(ROOT_DIR + '/submissions/' + file_name, '+w')
        f.write(test_header + '\n')
        f.flush()

        if gzipped:
            submission_df.to_csv(f, sep=',', mode='a', header=False, index=False, compression='gzip')
        else:
            submission_df.to_csv(f, sep=',', mode='a', header=False, index=False)

        # Close and save the file
        f.close()

        # Verify the correctness of the submission
        print('Verifying submission...', flush=True)
        if verify:
            errors = verify_submission(ROOT_DIR + '/data/challenge/challenge_set.json',
                                       ROOT_DIR + '/submissions/' + file_name)

            if errors == 0:
                print('Submission successfully saved in /submission!')
                print("No errors found. C'mon Creamy Fireflies!")
            else:
                print("Your submission has", errors, "errors. If you submit it, it will be rejected.")
                os.remove(ROOT_DIR + '/submissions/' + file_name)

