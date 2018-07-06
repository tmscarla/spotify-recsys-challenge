# Run Recommenders Guide

Follow this guide to rebuild from scratch all the matrices used in the final ensemble and in the clustered approach.

All the steps for both main and creative track need the **virtual environment** to be active, if you have not activated it yet, place yourself in the root folder of the project and run the following command:
> $ source py3e/bin/activate

If you want to re-generate our final CSV ready to be submitted, please make sure that every script is called in **online**
mode. Otherwise if you have filled the /data/test1 folder, and therefore you have our local validation set, you can call each script in **offline** mode for local testing.

### Main track

1. Rebuild the tuned matrices for each single algorithm.<br/>
The matrices created are built with the best parameters used for the final submission for every category. <br/>
The workflow to generate each matrix can be found in the relative python script.
    
    > $ cd recommenders/script/main
    
    > $ ./generate_all_npz.sh online
    
2. Then move to the /run folder of the project and launch the following script to generate the clustered matrices
    
    > $ python gen_clustered_matrices_main.py online
    
3. Still in the run folder, launch this script to ensemble all the matrices with their relative weights

    > $ python run_main_from_scratch_online.py

4. Retrieve the CSV file ready to be submitted in the /submissions folder
    

### Creative rack

0. Download our enriched data or recreate the enriched dataset with the following instruction.

    > $ python run/creative_data_collector.py <client_id> <client_secret>

1. Rebuild the tuned matrices of each single algorithm, which are trained also with external datasets.

    > $ python run/gen_creative_layered_matrix.py
    
    > $ recommenders/script/creative/generate_all_npz.sh online
     
2. Generate the clustered matrices from those ones

    > $ python run/gen_clustered_matrices_creative.py online
    
3. Ensemble all the matrices with their relative weights

    > $ python run/run_creative_from_scratch_online.py  
    
4. Retrieve the CSV file ready to be submitted in the /submissions folder
    
   

