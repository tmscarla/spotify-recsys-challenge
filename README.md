# Spotify Recsys Challenge 2018

## Challenge
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

## Data
In order to load data in an easy way, we converted the original JSON files provided from Spotify in CSV files.
Since data are not publicly available, we cannot include it in the repo.

There are two ways to include CSV files in the repo:
   * If you have **original JSON** files, run the following two scripts
   in order in the /run folder:
    
     > python mpd_to_csv.py path/to/mpd
     
     > python challenge_set_to_csv.py path/to/challenge_set.json
     
   * If you are a **challenge organizer**, you can send us an email at creamy.fireflies@gmail.com
    and we will provide you as soon as possible the entire /data folder which you can simply add to the repo
    avoiding to convert all the files.

Finally include the *challenge_set.json* file into /data/challenge folder.
 
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

## Setting up the environment:

1. Clone the repository on a machine running Ubuntu.
2. Run the following script to install the virtualenv, Python dependencies, the package of the repository
 and compile Cython code.
    > $ ./setup_ubuntu.sh
3. Activate the virtualenv to run any of the python script in the repository
    > source py3env/bin/activate

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
Once you have /data folder correctly filled with csv files, if you want to reproduce
our final submissions, move to /run folder and run the following scripts:

   * Main track
     > python run_main.py
   * Creative track
     > python run_creative.py

Once the computation is terminated, you should see the csv files ready to be submitted 
in the /submissions folder.

The previous scripts leverages on the fact that estimated user rating matrix for each algorithm have been previously computed.
If you want to run each recommender separately, please follow the instructions in the **[Run Recommenders Guide](https://github.com/tmscarla/spotify-recsys-challenge/tree/master/run)**. If you are a challenge organizer and you are encountering some problems, do not hesitate to contact us via mail.

## Metrics
Submissions are evaluated using three different metrics and final rankings will be computed
 by using the **[Borda Count](https://en.wikipedia.org/wiki/Borda_count)** election strategy.
 
# Preprocessing

# Algorithms

## Main track

## Creative track

# Ensemble

<p align="center">
  <img width="100%" src="https://github.com/tmscarla/spotify-recsys-challenge/blob/master/images/ensemble.png">
</p>

# Postprocessing
Once we computed our EURM (Estimated User Rating Matrix), we tried to improve our score leveraging on domain-specific patterns of the dataset. Here is a list of the most useful techniques that we have developed:

### GapBoost

### TailBoost

### AlbumBoost

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

