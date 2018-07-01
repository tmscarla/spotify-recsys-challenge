import multiprocessing 
import time
import pandas as pd
import numpy as np
import re
import spacy
from spacy.lang.en.lemmatizer import LOOKUP
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en.stop_words import unicode_literals

from tqdm import tqdm
df = pd.read_csv(filepath_or_buffer="playlists.csv", sep="\t", header=0, usecols=['pid', 'name'])
print("# ENG MODEL IMPORT")
nlp = spacy.load('en_core_web_lg')
tmp = df[df['name'].notnull()]
df2 = tmp['name'][0:1000]


print("# STOPWORD GENERATION")

nlp.vocab["playlist"].is_stop = True
nlp.vocab["music"].is_stop = True
nlp.vocab["random"].is_stop = True
nlp.vocab["mix"].is_stop = True
nlp.vocab["new"].is_stop = True
nlp.vocab["good"].is_stop = True
nlp.vocab["song"].is_stop = True
nlp.vocab["like"].is_stop = True
nlp.vocab["today"].is_stop = True
nlp.vocab["my"].is_stop = True
nlp.vocab["i"].is_stop = True
nlp.vocab["in"].is_stop = True
nlp.vocab["the"].is_stop = True
nlp.vocab["old"].is_stop = True
nlp.vocab["-pron-"].is_stop = True
nlp.vocab["'s"].is_stop = True
nlp.vocab["hit"].is_stop = True
nlp.vocab["old"].is_stop = True

for word in STOP_WORDS:
    nlp.vocab[str(word)].is_stop = True

print("# TITLE NORMALIZATION")
def normalize_name(name):
    name = name.lower()
    name = re.sub(r'(.)\1{3,}', r'\1{2}', name)
    return nlp.pipe(name)





normalized_title = []


normalized_title = []

for row in tqdm(df2):
    print(normalize_name(row))
	
print("# GENRE IMPORT")
genre_df = pd.read_json('genre.json', orient= 'values', convert_dates='genre')
genre_df.columns = ['genre']
genre = []
for row in tqdm(genre_df['genre']):
    genre.append(nlp(row))
   

print("# SIMILARITY ")
sim = []
t = []
g =[]
for i in tqdm(normalized_title):
    for j in (genre):
        sim.append(i.similarity(j))
        t.append(str(i))
        g.append(j)
        


d = {'playlist': t, 'genre': g, 'sim' : sim}
sim_df = pd.DataFrame(data=d)
sim_df[sim_df['playlist'] == 'throwback'].sort_values('sim', ascending=False)

print("# TOP K IN SIMILARITY")

sim_m = []
t = []
g =[]
n_genre = len(genre)
for i in tqdm(normalized_title):
    tmp = []
    for j in (genre):
        tmp.append(i.similarity(j))
    top_n = np.array(tmp).argsort()[-(200):][::-1]
    x = np.zeros(n_genre)
    x[top_n] = np.array(tmp)[top_n]   
    sim_m.append(list(x))


tmp = np.array(sim_m)

print(tmp)
