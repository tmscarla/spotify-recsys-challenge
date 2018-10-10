<p align="center">
  <img width="75%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/spotifychallenge.png">
  <img width="60%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/logo.png">
</p>

<br>

## The Challenge
The **[RecSys Challenge 2018](https://recsys-challenge.spotify.com)** is organized by Spotify, The University of Massachusetts, Amherst, and Johannes Kepler University, Linz.
The goal of the challenge is to develop a system for the task of automatic playlist continuation.
Given a set of playlist features, participants’ systems shall generate a list of recommended tracks that can be added to that playlist, thereby ‘continuing’ the playlist.

The challenge is split into two parallel challenge tracks. In the main track, teams can only use data that is provided through the Million Playlist Dataset,
while in the creative track participants can use external, public and freely available data sources to boost their system.

## Overview
This repository contains all the approaches that we developed in order to solve the two tracks of the challenge.
The rest of the document is organized in the following way:

   * **Setup:** gives an overview of the project structure and instructions on how to gather data and 
   setting up everything in order tu run our scripts.
   * **Preprocessing:** here we list which data structures we used and how we combined features of the dataset.
   * **Algorithms:** in this section we show the different algorithms which are behind each recommender and their performance.
   * **Ensemble:** a section entirely dedicated to how we build our final recommender starting from different algorithms.
   * **Postprocessing:** a set of postprocessing techniques which aim at increasing the prediction score by looking at the       domain of the problem.

## Team members
We are Creamy Fireflies, a group of MSc students from Politecnico di Milano which
took part in the Spotify RecSys Challenge 2018. These are the team members:

Main track:
* **[Sebastiano Antenucci](https://github.com/sebastianoantenucci)**
* **[Simone Boglio](https://github.com/bogliosimone)**
* **[Emanuele Chioso](https://github.com/EmanueleChioso)**
* **[Tommaso Scarlatti](https://github.com/tmscarla)**

Creative track:
* **[Ervin Dervishaj](https://github.com/edervishaj)**
* **[Shuwen Kang](https://github.com/JessicaKANG)**

We worked under the supervision of a PhD student, who helped us with the most complicated tasks and
suggested several state of the art approaches:

* **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)**

# Setup


## Setting up the environment:

1. Clone the repository on a machine running Ubuntu.
2. Run the following script to install the virtualenv, Python dependencies, the package of the repository
 and compile Cython code.
    > $ ./setup_ubuntu.sh
3. Activate the virtualenv to run any of the python script in the repository
    > $ source py3e/bin/activate


## Data Preprocessing 
In order to load data in an easy way, we converted the original JSON files provided from Spotify in CSV files.
Since data are not publicly available, we cannot include it in the repo.

Include the *challenge_set.json* file into /data/challenge folder.
 
There are two ways to include CSV files in the repo:
   * If you have **original JSON** files, run the following two scripts
   in order in the /run folder:
    
     > $ python mpd_to_csv.py    "absolute path to mpd/data folder with json files"
     
     > $ python challenge_set_to_csv.py   "absolute path to the directory containing challenge_set.json"
     
   * If you are a **challenge organizer**, you can send us an email at creamy.fireflies@gmail.com
    and we will provide you as soon as possible the entire /data folder which you can simply add to the repo
    avoiding to convert all the files.
 
After these steps you should have the /data folder organized as follows:
   * original
      * albums.csv
      * artists.csv
      * test_interactions.csv
      * test_playlists.csv
      * tracks.csv
      * train_interactions.csv
      * train_playlists.csv
   * challenge
      * challenge_set.json
   * enriched
   * test1

## Folders
Here you have an overview of the struct of the project root: 

* **recommenders**  - recommenders class
* **data**          - dataset csv files
* **utils**         - functions and helper classes
* **scripts**       - running scripts
* **results**       - offline evaluation scores
* **pytests**       - unit tests
* **personal**      - team member personal experiments
* **boosts**        - boosting algorithms used in postprocessing phase
* **bayesian_scikit** - scikit-learn bayesian optimizator
* **submissions** - csv files ready to be submitted
* **tune** - files for tuning on validation set

These main folders have a README.md that explains the structure of the package.

## Reproduce our final results
Once you have /data folder correctly filled with csv files, you can reproduce
our final submissions in a quick way starting from pre-computed estimated user rating matrices. You can find all of these matrices in the *final_npz_main* and *final_npz_creative* folder. Just run the following scripts:
 
##### Main track
> $ python run/run_main.py
##### Creative track
> $ python run/run_creative.py

Once the computation is terminated, you should see the csv files ready to be submitted 
in the /submissions folder.

If you don't want to use intermediate results, but starting from scratch, please follow the instructions in the **[Run Recommenders Guide](https://github.com/tmscarla/spotify-recsys-challenge/tree/master/run)**
in order to run each recommender separately. If you are a challenge organizer and you are encountering some problems, do not hesitate to contact us via mail.

## Metrics
Submissions are evaluated using three different metrics and final rankings will be computed
 by using the **[Borda Count](https://en.wikipedia.org/wiki/Borda_count)** election strategy.
 The three metrics are:
 
 * R-Precision
 * Normalized discounted cumulative gain
 * Recommended Songs clicks
 
 You can find a more detailed explanation of the metrics on the **[rules page](https://recsys-challenge.spotify.com/rules)** of the challenge. 
 
# Preprocessing
Once we converted the original JSON files into CSV, we gather data into matrices in order to perform recommendations:

* **URM** (User Rating Matrix): a matrix (playlists x tracks) with binary interactions.
* **ICM** (Item Content Matrix): a matrix (tracks x features) with artist id and album id as features.
* **UCM** (User Content Matrix): a matrix (playlists x features) with tokens extracted from titles as features.

# Algorithms
Here you can find a list of the main algorithms used to compute the final recommendation. Each algorithm outputs a EURM (Estimated User Rating Matrix) of shape (10K, 2.2M) where on the rows we have the 10K test playlists and on the columns all the unique tracks present in the *Million Playlist Dataset*.

* Personalized Top Popular
* Collaborative Filtering - Track based
* Collaborative Filtering - Playlist based
* Content Based Filtering - Track based
* Content Based Filtering - Playlist based
  * Track features
  * Playlist name

## Creative Track

We spent considerable e ort in trying to reconcile the tracks from the Million Playlist Dataset (MPD) provided by Spotify with those from external datasets but matching the name of the tracks and artists proved to be di cult and error-prone. Spotify Web API, on the other hand, being an API provided by Spotify itself, allowed us to retrieve for all tracks in MPD and in the Challenge Dataset following features: acousticnes, danceability, energy, instrumentalness, live- ness, loudness, speechiness, tempo, valence, popularity.

### Audio Feature Layered Content Based Filtering

The creative CBF is implemented with the following steps:

  1. Divide the tracks into 4 clusters with equal number of ele- ments, according to each feature. Take the loudness feature as an example, the clustering result is shown in Figure 4.
  2. Considering feature clusters as a 3rd dimension, split the dense ICM into 4 sparse layers. A loudness based layered ICM is illustrated in Figure 5.
  3. Concatenate 4 layers of sparse matrices horizontally in order to create a  nal sparsi ed ICM.
  4. Applying the CBF approach to the sparsi ed ICM, we can calculate a sub-artist-level track-track similarity.

<p align="center">
    <img src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/loudness_hist_clustered.png">  
</center>

# Ensemble
In order to take advantage of the diversity in the recommendations of all the N algorithms, we divided the results by category and we use them to compute, for each playlist, N sets of tracks scores such that the highest valued tracks will be recommended to that playlist.

<p align="center">
  <img width="100%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/ensemble.png">
</p>

# Postprocessing
Once we computed our EURM (Estimated User Rating Matrix), we tried to improve our score leveraging on domain-specific patterns of the dataset. Here is a list of the most useful techniques that we have developed:

### GapBoost
It is an heuristic which applies to playlists of categories 8 and 10 of the challenge set, where known tracks for each playlist
are distributed at random. Since known tracks are not in order, there exsist "gaps" between each pair of known tracks. 
We exploit this information by reordering our final prediction giving more information to those tracks which seemed to "fit"
better between all the gaps of the playlist.

The boost for each track is calculated as follows:

<p align="center">
  <img width="30%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/gapboost.png">
</p>

where S is a similarity matrix between track, g are the tracks to the left and to the right of the gap, and d is the length of the gap.
GapBoost improves in particular the R-Precision metric.

<p align="center">
  <img width="60%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/gb_diagram.png">
</p>


### TailBoost
We applied this technique to categories 5, 6, 7, 9 of the challenge set, where known tracks for each playlist are given in order.
The basic idea behind this approach is that the last tracks are the most informative about the "continuation" of a playlist, therefore
we boosted all the top tracks similar to the last known tracks, starting from the tail and proceding back to the head
with a discount factor.

TailBoost improves significantly Recommender Songs clicks and NDCG metrics.
The implementation of the TailBoost is available in the */boosts/tail_boost.py* file. 

### AlbumBoost
This approach leverages on the fact that some playlists are built collecting tracks in order from a specific album.
Therefore in categories 3, 4, 7 and 9, where known tracks for each playlist are given in order, we used this heuristic
to boost all the tracks from a specific album where the last two known tracks belong to the same album.
AlbumBoost improves the Recommender Songs clicks metric. 

### Artists Clusters

 ## Requirements
| Package                         | Version        |
| --------------------------------|:--------------:|  
| **scikit-learn**                |   >= 0.19.1    |   
| **numpy**                       |   >= 1.14      |   
| **scipy**                       |   >= 1.0.0     |   
| **pandas**                      |   >= 0.22.0    |  
| **tqdm**                        |   >= 4.19.5    |
| **spotipy**                     |   >= 2.4.4     |
| **bayesian-optimization**       |   >= 0.6.0     |
| **matplotlib**                  |   >= 2.0.2     |
| **psutil**                      |   >= 5.2.2     |
| **sklearn**                     |   >= 0.19.0    |
| **nltk**                        |   >= 3.2.4     |
| **deap**                        |   >= 1.1.2     |

