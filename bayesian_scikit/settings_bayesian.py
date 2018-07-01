from utils.definitions import dump_params_dict, ROOT_DIR
import os
import stat

path_simo = ROOT_DIR+'/npz_simo/'
path_altre = ROOT_DIR+'/npz_altre/'

this_configuration_name= 'example'

norm =          ['max', 'max',    'max',   'max',  'max',  'max', 'max',  'l1',   'max',  'l1']
target_metric =  ['ndcg', 'ndcg', 'ndcg', 'ndcg', 'ndcg',  'ndcg',  'ndcg', 'ndcg', 'ndcg', 'ndcg']

conf1 = ['nlp_fusion', 'top_pop']
conf2 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','cf_al', 'cf_ar', 'nlp_fusion','top_pop_album_cat2','top_pop_track_cat2']
conf3 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','nlp_fusion']
conf4 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub']
conf5 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub', 'nlp_fusion']
conf6 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub']
conf7 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','nlp_fusion']
conf8 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','nlp_fusion',  'hyb_j_main_cat8']
conf9 = ['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','nlp_fusion',  'cb_ib_cat9' ]
conf10 =['cb_ar','cb_al','cb_al_ar','cf_ib','cf_ub','nlp_fusion',  'hyb_j_main_cat10']
         
name_settings = [conf1,conf2,conf3,conf4,conf5,conf6,conf7,conf8,conf9,conf10]


file_locations_offline = {  'cb_ar':          path_simo+"cb_ar_offline.npz", 
                            'cb_al':          path_simo+"cb_al_offline.npz",
                            'cb_al_ar':       path_simo+"cb_al_ar_offline.npz",
                            'cf_ib':          path_simo+"cf_ib_offline.npz",
                            'cf_ub':          path_simo+"cf_ub_offline.npz",

                            'cf_tom_album':   path_simo+"cf_al_offline.npz",
                            'cf_tom_artist':  path_simo+"cf_ar_offline.npz",

                            'cf_ar':          path_simo+"cf_ar_offline.npz",
                            'cf_al':          path_simo+"cf_al_offline.npz",

                            'cf_ib_new':          path_simo+"cf_ib_offline.npz",
                            'cf_ub_new':          path_simo+"cf_ub_offline.npz",

                            'hyb_j_main_cat10': path_simo+'cb_layer_cat10_offline.npz',
                            'hyb_j_main_cat8': path_simo+'cb_layer_cat8_offline.npz',
                          
                            # from keplero
                            'top_pop':        path_altre+'top_pop.npz',
                          
                            'nlp':            path_altre+'nlp_offline.npz',
                            'nlp_fusion':     path_altre+'nlp_fusion_offline.npz',
        
                            'slim':           path_altre+'slim_offline.npz',

                            'als':            path_altre+'als_offline.npz',

                            ##### things for single categories:

                            'top_pop_album_cat2': path_altre+'top_pop_2_album_offline.npz',
                            'top_pop_pers2'     : path_altre+'top_pop_2_track_offline.npz',
                            'top_pop_track_cat2': path_altre+'top_pop_2_track_offline.npz',

                            'cb_ib_cat9'   :  path_altre+'cb_ib_cat9_offline.npz',
                            'cat9_rp3'    :  path_altre+'cat9_cf_ib_vecchio_offline.npz'

                            }



file_locations_online = { 'cb_ar':          path_simo + "cb_ar_online.npz",
                          'cb_al':          path_simo + "cb_al_online.npz",
                          'cb_al_ar':       path_simo + "cb_al_ar_online.npz",
                          'cf_ib':          path_simo + "cf_ib_online.npz",
                          'cf_ub':          path_simo + "cf_ub_online.npz",

                          'cf_tom_album':   path_simo + "cf_al_online.npz",
                          'cf_tom_artist':  path_simo + "cf_ar_online.npz",
                          'cf_ar':          path_simo + "cf_ar_online.npz",
                          'cf_al':          path_simo + "cf_al_online.npz",

                          'cf_ib_new':      path_simo + "cf_ib_online.npz",
                          'cf_ub_new':      path_simo + "cf_ub_online.npz",

                          'hyb_j_main_cat10': path_simo+'cb_layer_cat10_online.npz',
                          'hyb_j_main_cat8': path_simo+'cb_layer_cat8_online.npz',

                          # from keplero, dovrebbero esser per tutte le cat 
                          'top_pop':        path_altre + 'top_pop.npz',

                          'nlp':            path_altre + 'nlp_online.npz',
                          'nlp_fusion':     path_altre + 'nlp_fusion_online.npz',

                          'slim':           path_altre + 'slim_online.npz',

                          ##### things for single categories:
                          'cb_ib_cat9':     path_altre + 'cb_ib_cat9_online.npz',

                          'top_pop_album_cat2': path_altre + 'top_pop_2_album_online.npz',
                          'top_pop_track_cat2': path_altre + 'top_pop_2_track_online.npz',
                          'top_pop_pers2':      path_altre + 'top_pop_2_track_online.npz'


                            }


#### checks
assert len(norm)==10
assert len(target_metric)==10
for n in norm:
    assert n == 'max' or n=='l1' or n=='l2' or n=='box_max' or n=='box_l1' or n=='q_uni','check your normalization: "'+n+'"'

for metric in target_metric:
    assert metric == 'ndcg' or metric =='prec' or metric== 'sum' ,'check your metric: "'+metric+'"'

for config_arr in name_settings:
    for matrix_name in config_arr:
        assert matrix_name in file_locations_offline.keys(),"'"+matrix_name+"'"


############################## STOP HERE :) #####################################


if not os.path.exists(this_configuration_name):
    os.mkdir(this_configuration_name)


### creation run.sh
with open( ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/run_'+this_configuration_name+'.sh','w') as outfile:
    outfile.write("#!/usr/bin/env bash\n")
    outfile.write("source ../../py3env/bin/activate\n")
    for i in range(1,10):
        outfile.write("python ../bayesian_common_files/generic_bayesian.py "
                      +str(i)+" "+target_metric[i-1]+' '+norm[i-1]+' '+this_configuration_name+" & " )
    outfile.write("python ../bayesian_common_files/generic_bayesian.py "
                      +str(10)+" "+target_metric[9]+' '+norm[9]+' '+this_configuration_name)

st = os.stat(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/run_'+this_configuration_name+'.sh')
os.chmod(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/run_'+this_configuration_name+'.sh',
         st.st_mode | stat.S_IEXEC)

### creation run_ensemble_offline.sh and creation run_ensemble_online.sh
types = ['online','offline']
for type in types:
    with open( ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/ensemble_'+type+'_'+this_configuration_name+'.sh','w') as outfile:
        outfile.write("#!/usr/bin/env bash\n")
        outfile.write("source ../../py3env/bin/activate\n")
        outfile.write("python ../bayesian_common_files/ensemble.py "+ this_configuration_name+" "+type)

    st = os.stat(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/ensemble_'+type+'_'+this_configuration_name+'.sh')
    os.chmod(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/ensemble_'+type+'_'+this_configuration_name+'.sh',
             st.st_mode | stat.S_IEXEC)


### creation run.sh
with open( ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/print_params_'+this_configuration_name+'.sh','w') as outfile:
    outfile.write("#!/usr/bin/env bash\n")
    outfile.write("source ../../py3env/bin/activate\n")
    outfile.write("python ../bayesian_common_files/print_params.py "+this_configuration_name)

st = os.stat(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/print_params_'+this_configuration_name+'.sh')
os.chmod(ROOT_DIR + '/bayesian_scikit/' + this_configuration_name + '/print_params_'+this_configuration_name+'.sh',
         st.st_mode | stat.S_IEXEC)



# crating readable settings
with open(ROOT_DIR+'/bayesian_scikit/'+this_configuration_name+'/readable_settings_'+this_configuration_name+'.txt', 'w') as outfile:
    for i, riga in enumerate(name_settings):
        outfile.write("cat"+str(i+1)+"\t>\t"+str(riga)+'\n')


# saving file locations and the settings
dump_params_dict( name_settings,   name='name_settings',
          path=ROOT_DIR+'/bayesian_scikit/'+this_configuration_name+'/')

dump_params_dict( file_locations_offline,name='file_locations_offline',
          path=ROOT_DIR+'/bayesian_scikit/bayesian_common_files/')

dump_params_dict( file_locations_online, name='file_locations_online',
          path=ROOT_DIR+'/bayesian_scikit/bayesian_common_files/')



