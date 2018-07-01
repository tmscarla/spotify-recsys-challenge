import pandas as pd
from collections import Counter


# count function
def df_counter(df=None, featname=None):
    dfList = list(df[featname])
    countDict = Counter(dfList)

    # prepare output df
    df_ = pd.DataFrame()
    df_[featname] = list(countDict.keys())
    df_['n'] = list(countDict.values())
    # sort
    df_ = df_.sort_values(by=featname)
    return df_


# cluster function
def cluster(df=None, K=None, feat=None, verbose=False):
    df = df.dropna()
    df_ = df_counter(df=df, featname=feat)
    n = len(df) / K
    clusterDict = {}
    i = 0
    numtid = 0
    itemList = []
    total = 0
    for item, ntids in zip(df_[feat], df_['n']):
        if numtid < n:
            numtid = numtid + ntids
            itemList.append(item)
        else:
            itemList.append(item)
            clusterDict[i] = itemList
            if verbose: print('Cluster', i, 'done with', len(itemList), 'items')
            total = total + len(itemList)
            # reset
            itemList = []
            i = i + 1
            numtid = 0
    # last cluster
    clusterDict[i] = itemList
    if verbose: print('Cluster', i, 'done with', len(itemList), 'items')
    total = total + len(itemList)
    if verbose: print('Total num of items clustered', total, ', should be', len(df_))

    # prepare output df
    cluster_list = []
    feat_list = []
    for item in clusterDict.keys():
        for i in clusterDict[item]:
            cluster_list.append(item)
            feat_list.append(i)
    df_tmp = pd.DataFrame()
    df_tmp[feat + '_cluster'] = cluster_list
    df_tmp[feat] = feat_list
    # match cluster
    df_out = df.merge(df_tmp, left_on=feat, right_on=feat, how='inner')
    if verbose: print('Check final match:', len(df), len(df_out), 'should be equal.')

    return df_out