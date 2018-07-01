# Spotify-Challenge

## Setting up the environment:

1. Clone or download the repository on a machine running Ubuntu.
2. run "setup_ubuntu.sh" it will install the virtualenv, python dependencies, this reapository as package and compile cython code.
3. activate the virtualenv to run any of the python scripts in run folder
    > source py3env/bin/activate
    

## Folders
The structure is divided with the following scheme: 

* **recommenders**  - recommenders class
* **data**          - csv files
* **utils**         - functions and helper classes
* **scripts**       - running scripts
* **results**       - offline evaluation scores
* **pytests**       - unit tests
* **papers**        - list of useful papers
* **personal**      - personal experiments
* **boosts**        - boosting algorithms used in postprocessing phase
* **bayesian_scikit** - scikit-learn bayesian optimizator
* **spotify** - python scripts provided for dataset stats
* **submissions** - csv files ready to be submitted
* **tune** - files for tuning on validation set


These main folders have a README.md that explains the structure of the package.

## Data
In order to load data in an easy way, we converted the original JSON files in CSV files



