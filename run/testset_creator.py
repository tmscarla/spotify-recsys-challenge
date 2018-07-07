import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# import emoji
import gc
from utils.definitions import ROOT_DIR
from collections import OrderedDict
from utils.datareader import Datareader

def check_conditions( df, mean, std, error=(1.5,1.5)):
    """
    checks if the dataframe given is near has the duration similar to the one that we want to create
    similar == if its mean, std and number of emojis is at +/- error[] from it
    :param df:      dataframe to check
    :param mean:    target mean
    :param std:     target std
    :param error:
    :return:
    """
    target_mean = np.mean(df['num_tracks'])
    target_std = np.std(df['num_tracks'])
    if mean > (target_mean + error[0]) or mean < (target_mean - error[0]):
        print("error m ",mean,target_mean)
        return False
    if std > (target_std + error[1]) or std < (target_std - error[1]):
        print("error s ",std,target_std)
        return False
    return True



def get_random_df_constrained( source_df, num_of_pl, min_v, max_v, mean, std, errors=(1.5, 1.5)):
    """
    iterates until it creates a dataframe that satisfies the conditions.
    """
    seed = 0
    while True:
        df = source_df[((source_df['num_tracks']) >= min_v) & ((source_df['num_tracks']) <= max_v)].sample(
            n=num_of_pl, random_state=seed)
        if check_conditions(df, mean=mean, std=std, error=errors):
            break
        seed+=1
    return df,seed

def generate_train(playlists):

    ## mean
    cates = {'cat1': (10, 50, 1000, 28.6, 11.2), 'cat2_1': (10, 40, 998, 23.8, 8.7),
             'cat2_2': (70, 80, 2, 75, 4), 'cat3_1': (10, 50, 314, 29.4, 11.4),
             'cat3_2': (51, 75, 425, 62, 7.2), 'cat3_3': (75, 100, 261, 87, 7.1),
             'cat4': (40, 100, 1000, 63, 16.5), 'cat5': (40, 100, 1000, 63.5, 17.2),
             'cat6': (40, 100, 1000, 63.6, 16.7), 'cat7': (101, 250, 1000, 150, 38.6),
             'cat8': (101, 250, 1000, 151.7, 38.6), 'cat9': (150, 250, 1000, 189, 28),
             'cat_10': (150, 250, 1000, 187.5, 27)}
    cates = OrderedDict(sorted(cates.items(), key=lambda t: t[0]))
    cat_pids = {}

    seeds = [0] * len(cates)
    count = 0
    for cat, info in cates.items():
        print(cat)
        df, seeds[count] = get_random_df_constrained(playlists, min_v=info[0], max_v=info[1],
                                                           num_of_pl=info[2],
                                                           mean=info[3], std=info[4], errors=(1.5, 1.5))
        cat_pids[cat] = list(df.pid)
        playlists = playlists.drop(df.index)
        count += 1
    playlists = playlists.reset_index(drop=True)
    return playlists, cat_pids



def generate_test(cat_pids, playlists, interactions):

    def build_df_none(cat_pids, playlists, cat, num_samples):
        df = playlists[playlists['pid'].isin(cat_pids[cat])]
        df = df[['pid', 'num_tracks']]
        df['num_samples'] = num_samples
        df['num_holdouts'] = df['num_tracks'] - df['num_samples']
        return df

    def build_df_name(cat_pids, playlists, cat, num_samples):
        df = playlists[playlists['pid'].isin(cat_pids[cat])]
        df = df[['name', 'pid', 'num_tracks']]
        df['num_samples'] = num_samples
        df['num_holdouts'] = df['num_tracks'] - df['num_samples']
        return df

    df_test_pl = pd.DataFrame()
    df_test_itr = pd.DataFrame()
    df_eval_itr = pd.DataFrame()

    for cat in list(cat_pids.keys()):
        if cat == 'cat1':
            num_samples = 0
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            # all interactions used for evaluation
            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            print("cat1 done")

        elif cat == 'cat2_1' or cat == 'cat2_2':
            num_samples = 1
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)
            df_sample = df_itr[df_itr['pos'] == 0]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat2 done")

        elif cat == 'cat3_1' or cat == 'cat3_2' or cat == 'cat3_3':
            num_samples = 5
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]

            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat3 done")


        elif cat == 'cat4':
            num_samples = 5
            df = build_df_none(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat4 done")

        elif cat == 'cat5':
            num_samples = 10
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat5 done")

        elif cat == 'cat6':
            num_samples = 10
            df = build_df_none(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat6 done")

        elif cat == 'cat7':
            num_samples = 25
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat7 done")

        elif cat == 'cat8':
            num_samples = 25
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            for pid in cat_pids[cat]:
                df = df_itr[df_itr['pid'] == pid]
                df_sample = df.sample(n=num_samples)
                df_test_itr = pd.concat([df_test_itr, df_sample])
                df = df.drop(df_sample.index)
                df_eval_itr = pd.concat([df_eval_itr, df])

            print("cat8 done")

        elif cat == 'cat9':
            num_samples = 100
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            df_sample = df_itr[(df_itr['pos'] >= 0) & (df_itr['pos'] < num_samples)]
            df_test_itr = pd.concat([df_test_itr, df_sample])
            df_itr = df_itr.drop(df_sample.index)
            df_eval_itr = pd.concat([df_eval_itr, df_itr])

            print("cat9 done")

        elif cat == 'cat_10':
            num_samples = 100
            df = build_df_name(cat_pids, playlists, cat, num_samples)
            df_test_pl = pd.concat([df_test_pl, df])

            df_itr = interactions[interactions['pid'].isin(cat_pids[cat])]
            # clean interactions for training
            interactions = interactions.drop(df_itr.index)

            for pid in cat_pids[cat]:
                df = df_itr[df_itr['pid'] == pid]
                df_sample = df.sample(n=num_samples)
                df_test_itr = pd.concat([df_test_itr, df_sample])
                df = df.drop(df_sample.index)
                df_eval_itr = pd.concat([df_eval_itr, df])

            print("cat10 done")
        else:
            raise(Exception,"cat not present ")
            exit()

    tracks = pd.read_csv(ROOT_DIR+"/data/original/tracks.csv", delimiter='\t')
    tids = set(df_eval_itr['tid'])
    df = tracks[tracks['tid'].isin(tids)]
    df = df[['tid', 'arid']]
    df_eval_itr = pd.merge(df_eval_itr, df, on='tid')
    del (tracks)
    del (df)


    df_test_pl.reset_index(inplace=True, drop=True)
    df_test_itr.reset_index(inplace=True, drop=True)
    df_eval_itr.reset_index(inplace=True, drop=True)
    interactions.reset_index(inplace=True, drop=True)

    return df_test_pl, df_test_itr, df_eval_itr, interactions

def main_10k_without_seeds():
    df_playlists = pd.read_csv(ROOT_DIR+"/data/original/train_playlists.csv", delimiter='\t')
    df_playlists.drop(['num_tracks.1'], axis=1, inplace=True)

    interactions = pd.read_csv(ROOT_DIR+"/data/original/train_interactions.csv", delimiter='\t')

    df_train_pl, cat_pids = generate_train(df_playlists)
    df_test_pl, df_test_itr, df_eval_itr, df_train_itr = generate_test(cat_pids, df_playlists, interactions)

    del (df_playlists)
    del (interactions)

    return df_train_pl, df_test_pl, df_test_itr, df_eval_itr, df_train_itr



def write_files( df_train_pl, df_test_pl, df_test_itr, df_eval_itr, df_train_itr, foldername="" ):

    if not os.path.exists(ROOT_DIR+'/data/'+foldername):
        os.makedirs(ROOT_DIR+'/data/'+foldername)

    print("train playlists")
    df_train_pl.to_csv(ROOT_DIR+'/data/'+foldername+"/train_playlists.csv", sep='\t', index=False)
    del (df_train_pl)

    print("test playlists")
    df_test_pl.to_csv(ROOT_DIR+'/data/'+foldername+"/test_playlists.csv", sep='\t', index=False)
    del (df_test_pl)

    print("test interactions")
    df_test_itr.to_csv(ROOT_DIR+'/data/'+foldername+"/test_interactions.csv", sep='\t', index=False)
    del (df_test_itr)

    print("evaluation interactions")
    df_eval_itr.to_csv(ROOT_DIR+'/data/'+foldername+"/eval_interactions.csv", sep='\t', index=False)
    del (df_eval_itr)

    print("train interactions")
    df_train_itr.to_csv(ROOT_DIR+'/data/'+foldername+"/train_interactions.csv", sep='\t', index=False)


def reorder_test_playlists():
    # Playlists reordering
    dr = Datareader(test_num=1, mode='offline', only_load=True)
    path_test_csv = dr._Datareader__path + dr._Datareader__test_playlist_file
    pids = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pids.extend(dr.get_test_pids(cat=i))
    test_playlists_df = pd.read_csv(path_test_csv, sep='\t', encoding='utf-8')

    test_playlists_df = test_playlists_df.set_index(['pid'])

    # Load and resave csv file ordered by cat
    test_playlists_df = test_playlists_df.reindex(pids)
    test_playlists_df['pid'] = test_playlists_df.index
    test_playlists_df.to_csv(path_test_csv , sep='\t', index=False, encoding='utf-8')


if __name__ == '__main__':

    df_train_pl, df_test_pl, df_test_itr, df_eval_itr, df_train_itr = main_10k_without_seeds()

    write_files(df_train_pl=df_train_pl , df_test_pl=df_test_pl, df_test_itr=df_test_itr,
                df_eval_itr=df_eval_itr, df_train_itr= df_train_itr, foldername="test1" )

    reorder_test_playlists()
