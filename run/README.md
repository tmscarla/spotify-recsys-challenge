# Run Recommenders Guide

We setted up two different ways to run the project, the easyest and partial one and the complete one.

  1. Only replicate our results using the estimated user rating matrices. 
  
  2. Rebuild from scratch all the matrices used in the final ensemble and in the clustered approach, using bayesian parameters to ensemble them.
  
 
## 1. Run from ensembled matrices

This procedure only replicates our final results. <br/>
It ensembles a snapshot of the bayesian optimizations with the best parameters of our research.
You can find the matrices used in the *final_npz_main* and *final_npz_creative* folders. 
 
##### Main track
> python run/run_main.py
##### Creative track
> python run/run_creative.py

## 2. Rebuild from scratch using bayesian parameters
 
All the steps for both main and creative track need the **virtual environment** to be active, if you have not activated it yet,
place yourself in the root folder of the project and run the following command:
> $  source py3e/bin/activate

#### Main track

1. Rebuild the tuned matrices of the single algorithms.<br/>
The matrices created are already with the parameters used for the final submission for every category. <br/>
The processes for each matrix cam be found in the relative python script.

    > $ recommenders/script/main/generate_icm_layer.sh online
    
    > $ recommenders/script/main/generate_all_npz.sh online
    
2. Generate the clustered matrices from those ones

    > python run/gen_clustered_matrices_main.py online
    
3. Ensemble all the matrices with their relative weights

    > python run/run_main_from_scratch_online.py
    

#### Creative rack

1. Rebuild the tuned matrices of the single algorithms.<br/> The algorythms that are called use the enriched dataset.

    > $ recommenders/script/creative/generate_icm_layer.sh online
    
    > $ recommenders/script/creative/generate_all_npz.sh online
     
2. Generate the clustered matrices from those ones

    > python run/gen_clustered_matrices_creative.py online
    
3. Ensemble all the matrices with their relative weights

    > python run/run_creative_from_scratch_online.py   
    
   

