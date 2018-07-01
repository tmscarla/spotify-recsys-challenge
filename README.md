# Spotify-Challenge

## Team members
We are Creamy Firelies, a group of MSc students from Politecnico di Milano which
took part in the Spotify RecSys Challenge 2018.
These are the team members:
* **[Sebastiano Antenucci](https://github.com/sebastianoantenucci)**
* **[Simone Boglio](https://github.com/bogliosimone)**
* **[Emanuele Chioso](https://github.com/EmanueleChioso)**
* **[Tommaso Scarlatti](https://github.com/tmscarla)**

## Data
In order to load data in an easy way, we converted the original JSON files provided from Spotify in CSV files.
Since data are not publicly available, we cannot include it in the repo.

There are two ways to include CSV files in the repo:
   * If you have **original JSON** files:
      * Run the jupyter notebook in /run/mdp_to_csv.ipynb
      * Then run the script in /run
        > python challenge_set_to_csv.py path/to/challenge_set.json
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
    
## Setting up the environment:

1. Clone or download the repository on a machine running Ubuntu.
2. run "setup_ubuntu.sh" it will install the virtualenv, python dependencies, this reapository as package and compile cython code.
3. activate the virtualenv to run any of the python scripts in run folder
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

## Final results

## Reproduce results

## How it works



