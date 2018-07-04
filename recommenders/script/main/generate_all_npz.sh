#!/bin/sh
echo "START Matrices layer cat 8 and 10 created"
python gen_hybrid_matrix_cat8_cat10.py $1
echo "DONE Matrices layer cat 8 and 10 created"

echo "START CB_AL"
python cb_al.py $1 $2
echo "DONE CB_AL"

echo "START CB_AR"
python cb_ar.py $1 $2
echo "DONE CB_AR"


echo "START CB_AL_AR"
python cb_al_ar.py $1 $2
echo "DONE CB_AL_AR"


echo "START  CF_AL"
python cf_al.py $1 $2
echo "DONE CF_AL"

echo "START CF_AR"
python cf_ar.py $1 $2
echo "DONE CF_AR"

echo "START CF_AL_AR"
python cf_al_ar.py $1 $2
echo "DONE CF_AL_AR"

echo "START CF_UB"
python cf_ub.py $1 $2
echo "DONE CF_UB"

echo "START CF_IB"
python cf_ib.py $1 $2
echo "DONE CF_IB"

echo "START LAYER_CAT8"
python cb_layer_cat8.py $1 $2
echo "DONE LAYER_CAT8"

echo "START LAYER_CAT10"
python cb_layer_cat10.py $1 $2
echo "DONE LAYER_CAT10"

echo "START NLP_FUSION"
python nlp_fusion.py $1 $2
echo "DONE NLP_FUSION"
