import pandas as pd
import time
from src.Splitter import *


start = time.time()
print("Loading file:")
playlists = pd.read_csv("../spotify/playlists.csv", delimiter='\t')
interactions = pd.read_csv("../spotify/interactions.csv", delimiter='\t')
end = time.time()
print(str(end-start) + "s")

start = time.time()
print("Splitting dataset:")
train_playlists, train_interactions, test_playlists, test_interactions, \
    eval_interactions = split_dataset(playlists, interactions)
end = time.time()
print(str(end-start) + "s")

