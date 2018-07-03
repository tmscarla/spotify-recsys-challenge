# Bayesian optimization 

### ***Settings***
 The settings_bayesian python script will create a folder with the configuration and update the global dictionary 
 for the matrix location.
 
 base configuration MAIN:
 
 > 
 
    this_configuration_name = 'example'
    
    norm = ['max', 'max', 'max', 'max', 'max', 'max', 'max', 'l1', 'max', 'l1']
    target_metric = ['ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg']
    
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
     
 base configuration CREATIVE    
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


        
     
  
 You can choose:
  * Normalization [l1,l2,max]
  * Objective metric [ndcg, rprec, click, sum ]<br/>
  sum will maximize the sum of ndcg and rprec
  * Rhe algorithms to be used for each category<br/>
 Every matrix should appear in bot online and offline dictionaries.
 
 
 
### RUN
after the settings are run, a folder with the configuration will be created.
run the shell script inside the folder
> run_configurationName.sh

while the bayesian runs, you can create a matrix with the actual best params.
> ensemble_offline_configurationName.sh

when the bayesian has reached the optimal value, ensemble for the online
> ensemble_online_configurationName.sh

 
##### run multiple instances
If the matrix location is the same for every configuration, you can run as many bayesian optimization as you need.
If the path to the matrices is modified, the "ensemble_online" and "ensemble_offline" will use the new location

