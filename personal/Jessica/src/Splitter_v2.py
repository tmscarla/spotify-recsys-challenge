import pandas as pd


def generate_train(playlists):
    # define category range
    cates = {'cat1': (10, 50), 'cat2': (10, 78), 'cat3': (10, 100), 'cat4': (40, 100), 'cat5': (40, 100),
             'cat6': (40, 100),'cat7': (101, 250), 'cat8': (101, 250), 'cat9': (150, 250), 'cat10': (150, 250)}

    cat_pids = {}
    for cat, interval in cates.items():
        df = playlists[(playlists['num_tracks'] >= interval[0]) & (playlists['num_tracks'] <= interval[1])].sample(
            n=1000)
        cat_pids[cat] = list(df.pid)
        playlists = playlists.drop(df.index)

    playlists = playlists.reset_index(drop=True)

    return playlists, cat_pids

def generate_test(cat_pids, playlists, interactions, tracks):

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

        if cat == 'cat2':
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

        if cat == 'cat3':
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

        if cat == 'cat4':
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

        if cat == 'cat5':
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

        if cat == 'cat6':
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

        if cat == 'cat7':
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

        if cat == 'cat8':
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

        if cat == 'cat9':
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

        if cat == 'cat10':
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

    tids = set(df_eval_itr['tid'])
    df = tracks[tracks['tid'].isin(tids)]
    df = df[['tid', 'arid']]
    df_eval_itr = pd.merge(df_eval_itr, df, on='tid')

    df_test_pl = df_test_pl.reset_index(drop=True)
    df_test_itr = df_test_itr.reset_index(drop=True)
    df_eval_itr = df_eval_itr.reset_index(drop=True)
    interactions = interactions.reset_index(drop=True)  # return as train_interactions

    return df_test_pl, df_test_itr, df_eval_itr, interactions

def split_dataset(df_playlists, df_interactions, df_tracks):
    """
    Split the MPD according to Challenge_set features
    :param df_playlists: DataFrame from "playlists.csv"
    :param df_interactions: DataFrame from "interactions.csv"
    :param df_tracks: DataFrame from "tracks.csv"
    :return: df_train_pl: a DataFrame with same shape as "playlists.csv" for training
             df_train_itr: a DataFrame with same shape as "interactions.csv" for training

             df_test_pl: a DataFrame of 10,000 incomplete playlists for testing
             df_test_itr: a DataFrame with same shape as " interactions.csv" for testing

             df_eval_itr: a DataFrame of holdout interactions for evaluation
    """
    df_train_pl, cat_pids = generate_train(df_playlists)
    df_test_pl, df_test_itr, df_eval_itr, df_train_itr = generate_test(cat_pids, df_playlists, df_interactions, df_tracks)

    return df_train_pl, df_train_itr, df_test_pl, df_test_itr, df_eval_itr



