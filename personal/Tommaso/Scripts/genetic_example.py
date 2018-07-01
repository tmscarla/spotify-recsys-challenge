import numpy as np

from personal.Tommaso.NLP.GA_FeatureSelection import GA_FeatureSelection
from personal.Tommaso.NLP.NLP import NLP
from utils.datareader import Datareader
from utils.definitions import STOP_WORDS


def customRandomDistribution75_25():
    return np.random.choice([0, 0, 0, 1])

def customRandomDistribution50_50():
    return np.random.choice([0, 1, 0, 1])

def rand0_1():
    return np.random.uniform(0, 1)


dr = Datareader(mode='offline', only_load=True, verbose=False)
stopwords = STOP_WORDS
nlp = NLP(dr, stopwords=[])

# evaluator = Evaluator(dr)

ucm = nlp.get_UCM()
urm = dr.get_urm()
test_playlists = dr.get_test_pids()

GA = GA_FeatureSelection(ucm, urm, test_playlists, logFile='log.txt',
                         bestIndividualFile='top.txt',
                         mode="weighting", numGenerations=200, populationSize=90,
                         initialRandomDistribution=rand0_1)

GA.main()

