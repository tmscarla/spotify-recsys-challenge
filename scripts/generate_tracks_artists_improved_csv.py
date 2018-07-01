
# coding: utf-8

# # Execute this notebook first, after execute the notebook track_improved

# In[1]:
from utils.definitions import ROOT_DIR
filepath = ROOT_DIR+'/data/original/'
# file path
artist_csv = "artists.csv"
track_csv = "tracks.csv"
artist_improved_intermediate= "tracks_improved_intermediate.csv" # generated from previous notebook
artist_improved_intermediate_dict = "artists_improved_intermediate.csv" # generated from previous notebook
artist_improved_final = "tracks_improved.csv"
artist_improved_final_dict = "artists_improved.csv"


# In[2]:


#import
import pandas
import numpy as np
import scipy.sparse as sp
import re
import collections
import tqdm
import os


# In[3]:


# read data
df = pandas.read_csv(filepath_or_buffer=filepath+artist_csv,sep="\t",header=0,
                usecols=['arid','artist_name','artist_uri'],
                dtype={'arid':np.int32,'artist_name':str, 'artist_uri':str})
df = df [['arid','artist_name','artist_uri']]
df.head()


# In[4]:


#data
originals = df['artist_name'].values
artists = df['artist_name'].str.lower().values
arids = df['arid'].values
uris = df['artist_uri'].values
#print(arids.shape[0])


# In[5]:


# split in main and co-artists (1st level)
def reg(vect):
    exp = ''
    for s in vect:
        exp += s + '|'
    exp = exp [:-1]
    return exp


def split_main_co_artists(value, reg):
    values = re.split(reg,str(value))
    l = len(values)
    main = []
    co = []
    if l == 1:
        main.append(values[0])
    elif l == 2:
        main.append(values[0])
        co.append(values [1])
    else:
        main.append(values [0])
        for i in range(1,l):
            co.append(values[i])
    return main, co    

#regex
s = []
s.append('\s[\(\[]?featuring[\.\:\.\,]?\s')
s.append('\s[\(\[]?featurin[\.\:\.\,]?\s')
s.append('\s[\(\[]?featured[\.\:\.\,]?\s')
s.append('\s[\(\[]?starring[\.\:\.\,]?\s')
s.append('\s[\(\[]?feat[\.\:\.\,]?\s')
s.append('\s[\(\[]?ft[\.\:\.\,]?\s')
s.append('\s[\(\[]?aka[\.\:\.\,]?\s')
s.append('\s[\(\[]?-[\.\:\.\,]?\s')
s.append('\s[\(\[]?introducing[\.\:\.\,]?\s')
s.append('\s[\(\[]?presents[\.\:\.\,]?\s')
s.append('\s[\(\[]?present[\.\:\.\,]?\s')
s.append('\s[\(\[]?duet\swith[\.\:\.\,]?\s')
s.append('\s[\(\[]?with[\.\:\.\,]?\s')
s.append('\sw\/\s')
s.append('\sf\/\s')
s.append('\s?\/\s?')
s.append('\s?\,\s\&\s?')
s.append('\smeets?\s')
s.append('\sand\shis\s')
s.append('\sand\sher\s')
s.append('\sand\sthem\s')
s.append('\s\&\shis\s')
s.append('\s\&\sher\s')
s.append('\s\&\sthem\s')
s.append('\s\&amp\;?\s')
s.append('[(|)]')
s.append('[\[|\]]')
s.append('[\{|\}]')
#spanish cases
s.append('\scon\sla\s')
s.append('\sy\ssus?\s')
s.append('\sy\slos?\s')
s.append('\spresenta\:?\s')
s.append('\scon\s')
s.append('\shaz\s')
#other lang
s.append('\smit\s')
s.append('\savec\s')
s.append('perf\.\s')
s.append('\slyr\.\s')
s.append('\sdir\.\s')
#special cases
s.append('\sfrom\:\s')
s.append('\sed\.\s')
s.append('\s?members\sof\sthe\s')
s.append('\s?members?\sof\s')
s.append('\svol\.?\s')
s.append('\s_\s')
s.append('performed\sby\s')
s.append('\spresents')
s.append('\s\'presents\'')
s.append('\spresents...')
s.append('\spresents\:')
s.append('\sfeaturng\s')
s.append('\sfeat\,')
s.append('[\(\[]feat[\.\:\.\,]')
s.append('feat\.')

reg_main_co_artists = reg(s)

c=0
main_a = []
co_a = []
for a in artists:
    main, co = split_main_co_artists(a,reg_main_co_artists)
    main_a.append(main)
    co_a.append(co)
    if len(co) + len(main) > 1:
        c += 1

if(len(main_a) != len(co_a)):
    print("ERROR")
else:
    pass
    #print("DONE, found %d instances"%(c))


# In[6]:


#split artists 2nd level (split main artists and after split the co-artists )

def split_artists(value, reg):
    artists = re.split(reg,str(value))
    return artists    

#regex
s = []
s.append('\sand\s')
s.append('\svs\.?')
s.append('\s?\-?conducted\sby\s')
s.append('\s?directed\sby\s')
s.append('\s?arranged\sby\s')
s.append('\sx\s')
s.append('\s\&\sco\.')
s.append('\s\&\s')
s.append('\s?\;\s?')
s.append('\s?\,\s?')
s.append('\s?\+\s?')
#spanish 
s.append('\sy\s')

reg_split_artists = reg(s)

main_a2 = []
co_a2 = []

# main artists
c1 = 0
for l_a in main_a:
    new_l = []
    for a in l_a:
        mains = split_artists(a,reg_split_artists)
        new_l = new_l + mains
        if len(mains)>1:
            #print (mains)
            c1 = c1 + 1
    main_a2.append(new_l)

# co-artists
c2 = 0
for l_a in co_a:
    new_l = []
    for a in l_a:
        co = split_artists(a,reg_split_artists)
        new_l = new_l + co
        if len(co)>1:
            #print (co)
            c2 = c2 + 1
    co_a2.append(new_l)

if(len(main_a2) != len(co_a2)):
    print("ERROR")
else:
    pass
    #print("DONE, found %d instances (%d main artists, %d co-artists)"%(c1+c2,c1,c2))


# In[8]:


# class artist (attributes and a couple of utility methods)
class Artist:
    def __init__(self, original_artist, arid, uri, 
                 main_artists = [], co_artists = [], 
                 main_artists_ids = [], co_artists_ids = [], 
                ):
        self.arid = arid
        self.original_artist = original_artist
        self.main = main_artists
        self.co = co_artists
        self.main_ids = main_artists_ids
        self.co_ids = co_artists_ids
        self.uri = uri
        self.clean_names()
        self.shif_co_if_main_empty()
    def clean_names(self):
        self.main = list(map(str.strip, self.main))
        self.co = list(map(str.strip, self.co))
        self.main = list(filter(lambda s: s!='', self.main))
        self.co = list(filter(lambda s: s!='', self.co))
    def shif_co_if_main_empty(self):
        #shift first co in main if main is empty (happens when a name of the artist start with parenthesis)
        if len(self.co) != 0 and len(self.main) ==0:
            self.main.append(self.co[0])
            self.co = self.co[1:]
        # artist with no name, actually without filter single char happens just one time
        #if len(self.co) == 0 and len(self.main) == 0:
            #self.main.append('None')
    def reset_main_co_ids(self):
        self.main_ids = []
        self.co_ids = []
        


# In[9]:


# create the artist objects
final_artists = []
for i in range (0,len(main_a2)):
    original = originals[i]
    main = main_a2[i]
    co = co_a2[i]
    uri = uris[i]
    arid = arids[i]
    final_artists.append(Artist(original,arid,uri,main,co))


# In[10]:


# stat and search for attributes
def print_info_artist(a):
    print ("original: \t%s"%(a.original_artist))
    print ("main: \t\t%s"%(a.main))
    print ("co: \t\t%s"%(a.co))
    print ("main ids: \t%s"%(a.main_ids))
    print ("co ids: \t%s"%(a.co_ids))
    print ("id: \t\t%s"%(a.arid))
    print ("uri: \t\t%s"%(a.uri))
    return

# In[11]:


# build new ids for the artists

def get_new_id(name):
    global count
    if name not in new_dict:
        new_dict[name] = count
        count += 1
    return new_dict[name]

new_dict = {}
count = 0

for a in final_artists:
    a.reset_main_co_ids()
    for name in a.main:
        a.main_ids.append(get_new_id(name))
    for name in a.co:
        a.co_ids.append(get_new_id(name))


#print ('new dictionary: %d artist'%(count))


# In[12]:


#TODO: clean artist which name is a stop word
# like: orquesta, orchestra, friends, karaoke, co., chorus, etc... (look stat analysis at the end for more details)


# In[13]:


# write new data in a new csv 

artist_fields = ['arid','artist_uri','main_ids','co_ids','artist_name']#,'main_names','co_names']

full = []
for a in final_artists:
    row = []
    row.append(a.arid)
    row.append(a.uri)
    row.append(a.main_ids)
    row.append(a.co_ids)
    row.append(a.original_artist)
    #row.append(a.main)
    #row.append(a.co)
    full.append(row)

import csv
with open(filepath+artist_improved_intermediate, "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(full)
print (artist_improved_intermediate +" created")


# In[14]:


# build new dictionary usefull for the future work on extraction of artist in the tracks name
artist_fields = ['new_arid','new_artist_name']

inv_map = {v: k for k, v in new_dict.items()}

if len(inv_map)!=len(new_dict):
    print('ERROR conversion dictionary')


# In[15]:


## write dict in csv    
import csv
full = []
for i in range(0,len(inv_map)):
    row = []
    row.append(i)
    row.append(inv_map[i])
    full.append(row)

with open(filepath+artist_improved_intermediate_dict, "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(full)
print (artist_improved_intermediate_dict +" created")


#TODO now or in preprocessing, remove stopwords

# # Execute the artists_improved notebook first, after execute this notebook


import pandas
import numpy as np
import re
import collections
import tqdm
from utils.datareader import Datareader
import ast


# In[3]:


df = pandas.read_csv(filepath_or_buffer=filepath+track_csv,sep="\t",header=0,
                usecols=['tid','arid','alid','track_name'],
                dtype={'tid':np.int32,'arid':np.int32,'alid':np.int32,'track_name':str})
df = df [['tid','arid','alid','track_name']]
df.head()


# In[4]:


names = df['track_name'].str.lower().values
tids = df['tid'].values
alids = df['alid'].values
arids = df['arid'].values
#print('%d total tracks'%tids.shape[0])


# In[5]:


# get the full matrix (dataset + testset)
dr = Datareader(mode='online', only_load=True, verbose=False )
urm = dr.get_urm()
#print(urm.shape) 


# In[6]:


# just focus on songs that appear more than 1 time (-> threshold=2)
popularity = urm.sum(axis=0).A1
threshold = 0
ids_usefull_tracks = np.argwhere(popularity>=threshold)
#print('%d / %d usefull tracks (threshold >= %d)'%(ids_usefull_tracks.shape[0], popularity.shape[0], threshold))


# In[7]:


# class track
class Track:
    def __init__(self, tid, alid, arid, name):
        self.tid = tid
        self.alid = alid
        self.arid = arid
        self.name = name
        self.main_ar = []
        self.main_ar2 = []
        self.co_ar = []
        self.co_ar2 = []
        
# explore dataset function
def explore(string, n=10000):
    c=0
    for t in tracks[0:n]:
        if string in str(t.name):
            c+=1
            print(t.name)
    print('%d instances'%(c))

def explore_main(string, n=10000):
    c=0
    for t in tracks[0:n]:
        for a in t.main_ar:
            if string in str(a):
                c+=1
                print(str(a))
    print('%d instances'%(c))

def explore_co(string, n=10000):
    c=0
    for t in tracks[0:n]:
        for a in t.co_ar:
            if string in str(a):
                c+=1
                print(str(a))
    print('%d instances'%(c))


# In[8]:


# filter tracks above threshold and build objects
tracks = []
for index in ids_usefull_tracks:
    index = int(index) #leave this or you get array and no values
    new_track = Track(tids[index], alids[index], arids[index], names[index])
    tracks.append(new_track)
#print('%d objects Track created'%len(tracks))


# In[9]:


# split in main and co-artist (1st level)
def reg(vect):
    exp = ''
    for s in vect:
        exp += s + '|'
    exp = exp [:-1]
    return exp

def split_name(value, reg):
    values = re.split(reg,str(value))
    l = len(values)
    main = []
    co = []
    if l == 1:
        main.append(values[0])
    elif l == 2:
        main.append(values[0])
        co.append(values [1])
    else:
        main.append(values [0])
        for i in range(1,l):
            co.append(values[i])
    return main, co    

def remove_multiple_strings(cur_string, replace_list):
    for cur_word in replace_list:
        cur_string = cur_string.replace(cur_word, '')
    return cur_string


#replace list
r = []
r.append('remix')
r.append('explicit album version')
r.append('explicit version')
r.append('explicit')


#regex
s = []
#s.append('\(featuring\.?(.*?)\)')
#s.append('\(feat\.?(.*?)\)')
#s.append('\((.*?)\)')
s.append('[(|)]')
s.append('[\[|\]]')
s.append('[{|}]')
s.append('\s-\s')
#s.append('\sfeat\.?\s')

reg_names = reg(s)

c=0
main_a = []
co_a = []

n=len(tracks)

for t in tracks[0:n]:
    t.main_ar = []
    t.co_ar = []
    main, co = split_name(t.name,reg_names)
    if (len(co)>0):
        #print(str(main)+' % '+str(co))
        pass
    t.main_ar=main
    t.co_ar=co
    if len(t.main_ar)==0:
        print('ERROR splitting')
    if len(co) + len(main) > 1:
        c += 1

if(len(main_a) != len(co_a)):
    print("ERROR")
else:
    pass
    #print("DONE, found %d instances on %d total"%(c,n))


# In[10]:


# clear track names with feat and featuring no inside parenthesis

# split main name with no parenthesis
def split_artists(value, reg):
    values = re.split(reg,str(value))
    return values   


def clean_names(names):
        names = list(map(str.strip, names))
        names = list(filter(lambda s: s!='', names))
        return names
        
#split track name and artist(s)
s=[]
s.append('\sfeat[\.\:\.\,]?\s')
s.append('\sft[\.\:\.\,]?\s')
s.append('\sfeaturing[\.\:\.\,]?\s')

#split artists
w=[]
w.append('\s&\s')
w.append('\sand\s')
w.append('\,')
w.append('from')

regex = reg(s)
regex2 = reg(w)
counter=0
counter2=0
for t in tracks[0:n]:
    main, co = split_name(t.main_ar[0], regex)
    t.new_title = main[0]
    new_co=[]
    if (len(co)>0):
        counter+=1       
        for c in co:
            new_co+=split_artists(c,regex2)
        if len(new_co)>1:
            pass
            #print(new_co)
    t.new_ar1 = clean_names(new_co)
    counter2+=len(t.new_ar1)
#print('DONE, found %d instances (%d artist) on %d total'%(counter, counter2, n))


# In[11]:


# now the shittiest part, clear thing insides parenthesis
word_l=[]
word_l+=['feat','featuring','ft.']

w=[]
w.append('\s?ft\.?\s')
w.append('\s?featuring\.?\s')
w.append('\s?feat\.?\s')
w.append('\s?feat\.?\s?')
w.append('\s&\s')
w.append('\s\\\s')
w.append('\sand\s')
w.append('\s?from\s')
w.append('\s?with\s')
w.append('\s?extended remix\s?')
w.append('\s?extended version\s?')
w.append('\s?lp version\s?')
w.append('\s?album version\s?')
w.append('\s?version\s?')
w.append('\s?remix\s?')
w.append('\s?explicit\s?')
w.append('\s?radio mix\s?')
w.append('\s?radio edit\s?')
w.append('\s?a cappella\s?')
w.append('\s?originally performed by\s?')
w.append('\s?performed by\s?')
w.append('\s?originally by\s?')


w.append('\,')
#w.append('from')

regex = reg(w)
c=0
for t in tracks[0:n]:
    t.new_ar2 = []
    for a in t.co_ar:
        if any(xs in a for xs in word_l):
            new_ar = split_artists(a,regex)
            t.new_ar2 += clean_names(new_ar)
            c+=len(t.new_ar2)
            #print(t.new_ar2)
#print('DONE, %d artist extracted'%c)


# In[12]:


# merge the two list
c=0
for t in tracks:
    t.new_ar = []
    for a in t.new_ar1:
        if a not in t.new_ar:
            t.new_ar.append(a)
            c+=1
    for a in t.new_ar2:
        if a not in t.new_ar:
            t.new_ar.append(a)
            c+=1
#print('DONE, %d total artists extracted'%c)


# In[13]:


# start the conversion


# In[14]:


# read data
df2 = pandas.read_csv(filepath_or_buffer=filepath+artist_improved_intermediate,sep="\t",header=0,
                usecols=['arid','artist_name','main_ids','co_ids'],
                dtype={'arid':np.int32,'artist_name':str, 'main_ids':'O','co_ids':'O'})
df2 = df2 [['arid','main_ids','co_ids']]
df2.head()
arid = df2['arid'].values
mains = df2['main_ids'].values
cos = df2['co_ids'].values


# In[15]:


class Artist:
    def __init__(self, mains, cos):
        self.mains = mains
        self.cos = cos


# In[16]:


# create dictionary artist ids:      old_id-> new_ids 
n=arid.shape[0]
dic_old_new={}
for i in range(n):
    m = np.array(ast.literal_eval(mains[i]), dtype=np.int32).tolist()
    c = np.array(ast.literal_eval(cos[i]), dtype=np.int32).tolist()
    dic_old_new[arid[i]]=Artist(m,c)


# In[17]:


# read dict new artits:              new id -> name
df3 = pandas.read_csv(filepath_or_buffer=filepath+artist_improved_intermediate_dict,sep="\t",header=0,
                usecols=['new_arid','new_artist_name'],
                dtype={'new_arid':np.int32,'new_artist_name':str})
df3 = df3 [['new_arid','new_artist_name']]
df3.head()
new_arid = df3['new_arid'].values
new_name = df3['new_artist_name'].values


# In[18]:


# dict id->name and dict name->id
dict_id_name = {}
dict_name_id = {}
for i in range(new_arid.shape[0]):
    dict_id_name[new_arid[i]]=new_name[i]
    dict_name_id[new_name[i]]=new_arid[i]
#print(len(dict_id_name))
#print(len(dict_name_id))


# In[19]:


# now let's start the conversion
# i consider all the extracted artist as co_artist
counter_new_arid = max(dict_id_name)+1
miss=0
# add new artist in dictionary
for t in tracks:
    for a in t.new_ar:
        if a not in dict_name_id:
            miss+=1
            dict_name_id[a]=counter_new_arid
            dict_id_name[counter_new_arid]=a
            counter_new_arid+=1
#print(len(dict_id_name))
#print(len(dict_name_id))


# In[20]:


#finally add new ids to each track to co artist, that doesn't already appear in the main or co ids
c=0
c2=0
for t in tracks:
    t.new_main=dic_old_new[t.arid].mains.copy()
    t.new_co=dic_old_new[t.arid].cos.copy()
    for a in t.new_ar:
        id_a = dict_name_id[a]
        c2+=1
        if id_a not in t.new_main and id_a not in t.new_co:
            t.new_co.append(id_a)
            c+=1
#print('DONE, add %d instances (%d tot)'%(c,c2))




# In[21]:

print (artist_improved_intermediate+" removed")
os.remove(filepath+artist_improved_intermediate)
print (artist_improved_intermediate_dict+ " removed")
os.remove(filepath+artist_improved_intermediate_dict)


# save new dict artist
artist_fields = ['new_arid','new_artist_name']

## write dict in csv    
import csv
full = []
for i in range(0,max(dict_id_name)):
    row = []
    row.append(i)
    row.append(dict_id_name[i])
    full.append(row)

with open(filepath+artist_improved_final_dict, "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(full)
print (artist_improved_final_dict+" created")


# In[22]:


# save new track file

artist_fields = ['tid','old_arid','alid','new_main_arids','new_co_arids','name']

full = []

#add new tracks
for t in tracks:
    row = []
    row.append(t.tid)
    row.append(t.arid)
    row.append(t.alid)
    row.append(t.new_main)
    row.append(t.new_co)
    row.append(t.name)
    full.append(row)

import csv
with open(filepath+artist_improved_final, "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(full)
print (artist_improved_final+" created")

print ("Completed")


