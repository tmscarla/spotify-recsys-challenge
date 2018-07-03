# SUBMISSIONS folder

### ***Replicate our final recommendations***
 Here there are the two python scripts that ensemble the final matrices of our research.
 You can find the matrices used in the *final_npz_main* and *final_npz_creative* folders and can be rebuilt following the steps below. 
 
### Rebuild from scratch 
NB: all the steps needs the virtual environment to be active



#### Main track

1. Rebuild or download the tuned matrices of the tuned single algorithms
    
    > recommenders/script/main/generate_all_npz.sh online
    
    > recommenders/script/main/generate_icm_layer.sh online
    
    The matrices created are already with the parameters used for the final submission for every category. 
    The processes for each matrix cam be found in the relative python script.

2. Base ensemble matrix: <br/> Run the bayesian optimization inside folder bayesian-scikit with the configuration below. when it reaches a convergence, run the script "ensemble_online_configurationname.sh" to build the matrix

>  
    path_simo = ROOT_DIR+'/recommenders/script/main/online_npz/   ## the newly created matrices
    
    norm     ['max', 'max',    'max',   'max',  'max',  'max', 'max',  'l1',   'max',  'l1']
    target_metric =  ['ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg',  'ndcg',  'ndcg', 'ndcg', 'ndcg', 'ndcg']
    
    cat1       nlp_fusion, top_pop 
    cat2	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, cf_al, cf_ar, nlp_fusion, top_pop_album_cat2, top_pop_track_cat2
    cat3	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion
    cat4	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub
    cat5	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion
    cat6	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub
    cat7	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion
    cat8	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion, hyb_j_main_cat8
    cat9	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion, cb_ib_cat9
    cat10	   cb_ar, cb_al, cb_al_ar, cf_ib, cf_ub, nlp_fusion, hyb_j_main_cat10"

3. Run the python script to split the matrices into 4 clustered submatrices
    > gen_clustered_matrices_main.py
    
4. Cluster approach (AR1,AR2,AR3,AR4) : <br/> Run the bayesian optimization tool for all the four newly created folders sets of matrices with the same configuration as the previous step.
    
    When a run is finished, ensemble the matrix before to start the next bayesian becouse changing the folder location will affect the global bayesian settings.
    
    * cluster with only one artist
    >'/recommenders/script/main/online_npz/npz_ar1/' 
    
    * cluster with low variance of artists
    log(#uniqueTracks / #uniqueArtists) < 1
    >'/recommenders/script/main/online_npz/npz_ar2/'    
        
    * cluster with medium variance of artists
    log(#uniqueTracks / #uniqueArtists) < 2 
    >'/recommenders/script/main/online_npz/npz_ar3/'
    
    * cluster with high variance of artists
    log(#uniqueTracks / #uniqueArtists) >= 2
    >'/recommenders/script/main/online_npz/npz_ar4/'
    

5. Run the run_main replacing the variables [ar1, ar2, ar3, ar4, ensemble] with a load to your matrices
    

   
#### Creative Track  TODO

1. download or recreate the enriched dataset.
    > python creative_data_collector.py <client_id> <client_secret>
    
2. create the basic matrices 
    > python gen_creative_layered_matrix.py
    
    > recommenders/script/creative/generate_all_npz.sh online 
    
3. Base ensemble matrix: <br/> Run the bayesian optimization inside folder bayesian-scikit with the configuration below. when it reaches a convergence, run the script "ensemble_online_configurationname.sh" to build the matrix
>  
    path_jess = ROOT_DIR+'/recommenders/script/creative/online_npz/   ## the newly created matrices
    
    norm     ['max', 'max',    'max',   'max',  'max',  'max', 'max',  'l1',   'max',  'l1']
    target_metric =  ['ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg',  'ndcg',  'ndcg', 'ndcg', 'ndcg', 'ndcg']
    
    conf1 = ['nlp_fusion', 'top_pop']
    conf2 = ['cf_ib','cb_ar', 'cb_al','cb_al_ar','cf_ub','cf_ar','cf_al','nlp_fusion','top_pop_album_cat2','top_pop_track_cat2',
                'cr_cb_ar']
    conf3 = ['cf_ib','cb_ar', 'cb_al','cb_al_ar','cf_ub', 'nlp_fusion',
                'cr_cb_ar']
    conf4 = ['cf_ib','cb_ar', 'cb_al','cb_al_ar','cf_ub',
                'cr_cb_ar']
    conf5 = ['cf_ib','cb_ar', 'cb_al','cb_al_ar','cf_ub', 'nlp_fusion',
                'cr_cb_ar']
    conf6 = ['cf_ib','cb_ar','cb_al','cb_al_ar','cf_ub',
                'cr_cb_ar']
    conf7 = ['cf_ib','cb_ar','cb_al','cb_al_ar','cf_ub', 'nlp_fusion',
                'cr_cb_ar']
    conf8 = ['cf_ib','cb_ar','cb_al','cb_al_ar','cf_ub', 'nlp_fusion', 'hyb_j_main_cat8',
                'cr_cb_ar', 'cr_cluster_creative']
    conf9 = ['cf_ib','cb_ar','cb_al','cb_al_ar','cf_ub', 'nlp_fusion', 'cb_ib_cat9' ,
                'cr_cb_ar']
    conf10 =['cb_al','cf_ib','cf_ub', 'hyb_j_main_cat10',
                'cr_cb_ar', 'cr_cb_al_ar', 'cr_cf_feats_cat10', 'cr_cluster_creative', 'cr_cf_ar']



    
