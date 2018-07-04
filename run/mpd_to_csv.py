
# coding: utf-8

# In[63]:


import json
import os
from pprint import *
from tqdm import tqdm
from utils.definitions import ROOT_DIR
import sys

arg = sys.argv[1:]
path_load = arg[0] # mpd data folder
#path_load = "mpd.v1/data/" #json folder

path_save = ROOT_DIR + "/data/original/" #where to save csv


# In[74]:


playlist_fields = ['pid','name', 'collaborative', 'modified_at', 'num_albums', 'num_tracks', 'num_followers',
'num_tracks', 'num_edits', 'duration_ms', 'num_artists','description']
### care, the description field is optional

track_fields = ['tid', 'arid' , 'alid', 'track_uri', 'track_name', 'duration_ms']

album_fields = ['alid','album_uri','album_name']

artist_fields = ['arid','artist_uri','artist_name']

interaction_fields = ['pid','tid','pos']

interactions = []
playlists = []
tracks = []
artists = []
albums = []

count_files = 0
count_playlists = 0
count_interactions = 0
count_tracks = 0
count_artists = 0
count_albums = 0
dict_tracks = {}
dict_artists = {}
dict_albums = {}


def process_mpd(path):
    global count_playlists
    global count_files
    filenames = os.listdir(path)
    for filename in tqdm(sorted(filenames)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice['info'])
            for playlist in mpd_slice['playlists']:
                process_playlist(playlist)
                pid = playlist['pid']
                for track in playlist['tracks']:
                    track['pid']=pid
                    new = add_id_artist(track)
                    if new: process_artist(track)
                    new = add_id_album(track)
                    if new: process_album(track)
                    new = add_id_track(track)
                    if new: process_track(track)
                    process_interaction(track)
                count_playlists += 1
            count_files +=1

    show_summary()
    
def process_info(value):
    #print (json.dumps(value, indent=3, sort_keys=False))
    pass

def add_id_track(track):
    global count_tracks
    if track['track_uri'] not in dict_tracks:
        dict_tracks[track['track_uri']] = count_tracks
        track['tid'] = count_tracks
        count_tracks += 1
        return True
    else:
        track['tid'] = dict_tracks[track['track_uri']]
        return False

def add_id_artist(track):
    global count_artists
    if track['artist_uri'] not in dict_artists:
        dict_artists[track['artist_uri']] = count_artists
        track['arid'] = count_artists
        count_artists += 1
        return True
    else:
        track['arid'] = dict_artists[track['artist_uri']]
        return False

def add_id_album(track):
    global count_albums
    if track['album_uri'] not in dict_albums:
        dict_albums[track['album_uri']] = count_albums
        track['alid'] = count_albums
        count_albums += 1
        return True
    else:
        track['alid'] = dict_albums[track['album_uri']]
        return False

def process_track(track):
    global track_fields
    info = []
    for field in track_fields:
        info.append(track[field])
    tracks.append(info)

def process_album(track):
    global album_fields
    info = []
    for field in album_fields:
        info.append(track[field])
    albums.append(info)

def process_artist(track):
    global artist_fields
    info = []
    for field in artist_fields:
        info.append(track[field])
    artists.append(info)

def process_interaction(track):
    global interaction_fields
    global count_interactions
    info = []
    for field in interaction_fields:
        info.append(track[field])
    interactions.append(info)
    count_interactions +=1

def process_playlist(playlist):
    global playlist_fields
    if not 'description' in playlist:
        playlist['description'] = None
    info = []
    for field in playlist_fields:
        info.append(playlist[field])
    playlists.append(info)
    
    
        
def show_summary():
    print (count_files)
    print (count_playlists)
    print (count_tracks)
    print (count_artists)
    print (count_albums)
    print (count_interactions)


# In[66]:


process_mpd(path_load)


# In[67]:


import csv

with open(path_save+"artists.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(artists)
print ("artists.csv done")

with open(path_save+"albums.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(album_fields)
    writer.writerows(albums)
print ("albums.csv done")
    
with open(path_save+"interactions.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(interaction_fields)
    writer.writerows(interactions)
print ("interactions.csv done")

with open(path_save+"tracks.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(track_fields)
    writer.writerows(tracks)
print ("tracks.csv done")

with open(path_save+"playlists.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(playlist_fields)
    writer.writerows(playlists)
print ("playlists.csv done")

