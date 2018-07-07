
## Bayesian Optimization Guide

### Main Track

1. Generate the validation_set from**run folder**
    
    > python testset_creator.py
    
2. Generate all the matrices for both online and offline from**recommenders/script/main folder**

    > $ ./generate_all_npz.sh online
    
    > $ ./generate_all_npz.sh offline

3. Run the python script to split the matrices into 4 clustered submatrices from**run folder**
    
    > $ python gen_clustered_matrices_main.py online
    
    > $ python gen_clustered_matrices_main.py offline
   

3. Bayesian optimization for Base ensemble matrix: <br/> Run the bayesian optimization inside folder bayesian-scikit with the configuration below. when it reaches a convergence, run the script "ensemble_online_configurationname.sh" to build the matrix
   <br/>The search for the best parameter should be done for the OFFLINE, then it will use the same parameters for the ONLINE.
   In the *bayesian_scikit* folder there is an example of this ensemble
   <br/><br/>
   The number of tries is proportional to the number of matrix for each category, it will stop at ~800 for the more complex ones and it will take ~4h. Im most of the cases, after 300 tries it will reach the convergence.
    
    The folling commands work with the settings below.
   
        $ cd bayesian_scikit
        python settings_bayesian.py
        $ cd exampleConfig
        $ ./run_exampleConfig.sh
    
    Settings inside "settings_bayesian.py"
            
            this_configuration_name = 'exampleConfig' 
            
            path_simo = ROOT_DIR+'/recommenders/script/main/online_npz/   ## the newly created matrices
            
            norm =            ['max', 'max',   'max',   'max',  'max',  'max', 'max',  'l1',   'max',  'l1']
            target_metric =  ['ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg',  'ndcg',  'ndcg', 'ndcg', 'ndcg', 'ndcg']
            
            conf1 = ['nlp_fusion', 'top_pop']
            conf2 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'cf_al', 'cf_ar', 'nlp_fusion', 'top_pop_album_cat2', 'top_pop_track_cat2']
            conf3 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion']
            conf4 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub']
            conf5 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion']
            conf6 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub']
            conf7 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion']
            conf8 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion', 'hyb_j_main_cat8']
            conf9 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion', 'cb_ib_cat9']
            conf10 = ['cb_ar', 'cb_al', 'cb_al_ar', 'cf_ib', 'cf_ub', 'nlp_fusion', 'hyb_j_main_cat10']   
    
4.  You can now create the online and offline ensemble matrix while the bayesian is running with the current best parameters. 
        
        $ ./ensemble_offline_exampleConfig.sh
        $ ./ensemble_online_exampleConfig.sh
       

    
5. Cluster approach (cluster1,cluster2,cluster3,cluster4) : <br/> Run the bayesian optimization tool for all the four folders created by sets of matrices with the same configuration as the previous step.
    
    The settings now must point to  the folder of the clustered matrics. 
        
    Normalizations, objective function and category2 to category10 like the base example

        this_configuration_name = 'clusterAR1' 
        
        path_simo = ROOT_DIR+'/recommenders/script/main/online_npz/npz_ar1   ## the newly created matrices
       
        
        conf1 = ['nlp_fusion']
    
    When a run is finished, ensemble the matrix before to start the next bayesian becouse changing the folder location will affect the global bayesian settings.
        
        $ ./ensemble_offline_clusterAR1.sh
        $ ./ensemble_online_clusterAR1.sh
        
6. Repeat step 5 for all the 3 other clusters    

    * cluster with low variance of artists
    log(#uniqueTracks / #uniqueArtists) < 1
        >'/recommenders/script/main/online_npz/npz_ar2/'    
        
    * cluster with medium variance of artists
    log(#uniqueTracks / #uniqueArtists) < 2 
        
        >'/recommenders/script/main/online_npz/npz_ar3/'
    
    * cluster with high variance of artists
    log(#uniqueTracks / #uniqueArtists) >= 2
        >'/recommenders/script/main/online_npz/npz_ar4/'
    

5. Run the run_main replacing the variables [cluster1, cluster2, cluster3, cluster4, ensemble] with a load to your new matrices
