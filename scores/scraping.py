from bs4 import BeautifulSoup
from urllib.request import urlopen
import os.path
from urllib.error import HTTPError
from urllib.error import URLError
import pandas as pd
from utils.definitions import ROOT_DIR
import numpy as np
from utils.bot import Bot_v1

print("SCREEEEEPOOOOOOO ")

pd.set_option('precision',6)

def clean(results ):
    if results['RPREC'].dtype == object:
        results['RPREC'] = pd.to_numeric(results['RPREC'].str.replace(' ', ''), errors='force')
    if results['NDCG'].dtype == object:
        results['NDCG'] = pd.to_numeric(results['NDCG'].str.replace(' ', ''), errors='force')
    if results['CLICKS'].dtype == object:
        results['CLICKS'] = pd.to_numeric(results['CLICKS'].str.replace(' ', ''), errors='force')

    results = results.sort_values(by=['Team', 'DATE'])
    results = results.round(6)
    results.drop_duplicates(inplace=True)
    return results

def filter_line(table_line):
    name = table_line[2]
    rprec = table_line[3]
    ndcg = table_line[5]
    clicks = table_line[7]
    date = table_line[11]
    return name, rprec, ndcg, clicks, date

try:
    main = urlopen("https://recsys-challenge.spotify.com/leaderboard")
    creative = urlopen("https://recsys-challenge.spotify.com/leaderboard/creative")

except HTTPError as e:
    print(e)

except URLError:
    print("Server down or incorrect domain")

else:
    main_res = BeautifulSoup(main.read(), "html5lib")
    creative_res = BeautifulSoup(creative.read(), "html5lib")

    pages = {'main': main_res, 'creative': creative_res }

    #for both the main and creative
    for type, page in pages.items():

        scores= []
        tags = page.findAll("tr")

        for raw_score in tags :
            line = raw_score.getText().replace(" ","").split('\n')
            name,rprec,ndcg,clicks,date = filter_line(line)

            if name!='Team' and name!="paul'stestteam":
                scores.append({ 'Team':name,
                                'RPREC':rprec,
                                'NDCG':ndcg,
                                'DATE':date ,
                                'CLICKS':clicks})

        results = pd.DataFrame(scores)
        results = results[['Team', 'RPREC',"NDCG","CLICKS","DATE"]]

        results = clean(results)

        results.to_csv(path_or_buf=ROOT_DIR+"/scores/"+type+"/actual_"+type+'.csv',
                           sep="\t", header=['Team','RPREC',"NDCG","CLICKS","DATE"], index=False)

        if os.path.isfile(ROOT_DIR+"/scores/"+type+"/"+type+".csv"):

            old = pd.read_csv(filepath_or_buffer=ROOT_DIR+"/scores/"+type+"/"+type+'.csv',sep="\t",
                        usecols=['Team','RPREC',"NDCG","CLICKS","DATE"],
                        dtype={'Team': str, 'RPREC': np.float32, 'NDCG': np.float32, "CLICKS": np.float32, "DATE": str})
            old = clean(old)

            updated = pd.concat([old, results]).drop_duplicates(keep=False).reset_index(drop=True)  # remove duplicates

            updated = updated.sort_values(by=['Team', 'DATE'])
            updated = clean(updated)

            updated = updated.round(6)
            updated.to_csv(path_or_buf=ROOT_DIR + "/scores/" + type + "/" + type + '.csv',
                           sep="\t", header=['Team', 'RPREC', "NDCG", "CLICKS", "DATE"], index=False)

            # search for the new ones, and send a notification
            common = results.merge(old, on=['Team', 'DATE'])
            new_ones = results[(~results.Team.isin(common.Team)) & (~results.Team.isin(common.DATE))]
            new_ones = clean(new_ones)

            teams_that_submitted = new_ones['Team'].as_matrix()
            if len(teams_that_submitted)>0:
                bot = Bot_v1(name=type+" LEADERBOARD UPDATE", chat_id="-1001356251815")
                names = ', '.join(teams_that_submitted)
                bot.send_message(text=names)
            print(type,"tutto a posto")

print("fine totale")
