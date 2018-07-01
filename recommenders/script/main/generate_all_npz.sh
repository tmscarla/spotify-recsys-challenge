#!/bin/sh

python cb_al.py $1 $2
echo "CB_AL"
python cb_ar.py $1 $2
echo "CB_AR"
python cb_al_ar.py $1 $2
echo "CB_AL_AR"
python cf_al.py $1 $2
echo "CF_AL"
python cf_ar.py $1 $2
echo "CF_AR"
python cf_al_ar.py $1 $2
echo "CF_AL_AR"
python cf_ub.py $1 $2
echo "CF_UB"
python cf_ib.py $1 $2
echo "CF_IB"
python cb_layer_cat8.py $1 $2
echo "LAYER_CAT8"
python cb_layer_cat10.py $1 $2
echo "LAYER_CAT10"
python nlp_fusion.py $1 $2
echo "NLP_FUSION"
