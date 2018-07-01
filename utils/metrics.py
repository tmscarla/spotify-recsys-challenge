import numpy as np
import scipy as sc
import pandas as pd
from math import log2, log10
from collections import Counter
import math
import time

"""
All metrics will be evaluated at both the track level (exact track must match) 
and the artist level (any track by that artist is a match).
"""


def timing(f):
    """
    Compute the execution time for function f.
    Add "@timing" on the top of the function to display it.
    """
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


def r_precision(prediction_t, test_t, prediction_a, test_a):
    """
    Compute the R Precision between the prediction and the test set.
    :param prediction_t: a numpy array of recommended tracks
    :param test_t: a numpy array of hold-out tracks
    :param prediction_a: a numpy array of recommended artists
    :param test_a: a numpy array of hold-out artists
    :return: prec_overall, prec_artist: tracks + artists precision, precision only from artist level
    """

    # Check parameters
    assert len(prediction_t) == len(prediction_a)

    # Initialization
    prec_overall = 0.0
    prec_artists = 0.0
    cut_off = len(test_t)

    # Compute track hits and mask for misses
    tracks_hits = list(((Counter(prediction_t[:cut_off]) & Counter(test_t)).elements()))
    mask = np.in1d(prediction_t[:cut_off], tracks_hits, invert=True)

    # Artists of tracks misses
    artists_left = prediction_a[:cut_off][mask]

    # Update precision for track hits
    prec_overall += float(len(tracks_hits))

    # For tracks misses, evaluate artist level
    for a in artists_left:
        if a in test_a:
            prec_overall += 0.25
            prec_artists += 0.25
            test_a = list(filter(lambda x: x != a, test_a))

    return prec_overall/len(test_t), prec_artists/len(test_t)


def r_precision_old(prediction, test):
    """
    Compute the R Precision between the prediction and the test set.
    :param prediction: a numpy array of recommended tracks/artists
    :param test: a numpy array of hold-out tracks/artists
    :return: precision: the number of tracks in common between prediction and test
    """
    # Compute the number of relevant tracks/artists in the prediction
    precision = np.array(list(((Counter(prediction[:len(test)]) & Counter(test)).elements()))).size / len(test)

    return precision


def dcg(prediction, test):
    """
    Compute the DCG, which measures the ranking quality of the recommended tracks/artists.
    :param prediction: a numpy array of recommended tracks/artists
    :param test: a numpy array of hold-out tracks
    :return: dcg: discounted cumulative gain
    """

    # Initialize dcg
    dcg = 0

    # Indices to take into account artists duplicates in recommendation
    test_i = 0
    pred_i = 0

    # Count num correct tracks/artits
    num_correct = 0

    # Sort prediction and test
    pred_argsorted = np.argsort(prediction, kind='mergesort')
    prediction = np.sort(prediction)
    test = np.sort(test)

    # Iterates on both test and prediction
    while pred_i < len(prediction) and test_i < len(test):

        if prediction[pred_i] < test[test_i]:
            pred_i += 1
        elif prediction[pred_i] > test[test_i]:
            test_i += 1
        else:
            if pred_argsorted[pred_i] == 0:
                dcg += 1
            else:
                dcg += 1 / log2(pred_argsorted[pred_i]+1)

            pred_i += 1
            test_i += 1
            num_correct += 1

    return dcg, num_correct


def dcg_sliced(prediction, test):
    """
    Compute the DCG sliced on the first [0:num_holdouts] tracks.

    :param prediction: a numpy array of recommended tracks/artists
    :param test: a numpy array of hold-out tracks
    :return: dcg: discounted cumulative gain
    """

    # Initialize dcg
    dcg = 0.
    num_correct = 0

    for i in range(len(test)):
        if np.isin(prediction[i], test):
            num_correct += 1

            if i == 0:
                dcg += 1
            else:
                dcg += 1. / log2(i + 1)

    return dcg, num_correct


def idcg(num_holdouts):
    """
    Compute the ideal DCG in which the relevant tracks are perfectly ranked.
    :param num_holdouts: total number of relevant tracks
    :return: idcg: ideal discounted cumulative gain
    """
    idcg = 1

    for i in range(1, num_holdouts):
        idcg += 1 / log2(i + 1)

    return idcg


def ndcg(prediction, test, is_sliced=False):
    """
    Compute the normalized DCG.
    IMPORTANT: now is_sliced = False should be the correct version.

    :param prediction: a numpy array of recommended tracks/artists
    :param test: a numpy array of hold-out tracks/artists
    :param is_sliced: compute the dcg only on the first [0:num_holdouts] tracks, otherwise on all the prediction
    :return: ndgc: dcg / idgc
    """
    if is_sliced:
        dcg_value, num_correct = dcg_sliced(prediction, test)
    else:
        dcg_value, num_correct = dcg(prediction, test)

    num_holdouts = len(test)
    idcg_value = idcg(num_holdouts)

    ndcg = dcg_value / idcg_value

    return ndcg


def recommended_songs_clicks(prediction, test):
    """
    Compute the Recommended Songs clicks metric.
    :param prediction: a numpy array of recommended tracks/artists
    :param test: a numpy array of hold-out tracks/artists
    :return: clicks: is the number of refreshes needed before a relevant track is encountered
    """
    clicks = 51

    intersect = np.in1d(prediction, test)

    for i in range(0, len(prediction)):
        if intersect[i]:
            clicks = math.floor(i/10)
            break
    return clicks

