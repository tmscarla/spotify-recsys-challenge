#!/bin/sh

python cr_cb_al_ar_bm25.py $1 $2
echo "cr_CB_AL_AR"

python cr_cb_ar_bm25.py $1 $2
echo "cr_CB_AR"

python cr_cf_ar_bm25.py $1 $2
echo "cr_CF_AR"

python cr_cf_feats_bm25.py $1 $2
echo "cr_CF_FEATS"

python cr_cf_feats_cat10_bm25.py $1 $2
echo "cr_CF_FEATS_CAT10"

python cr_cf_hyb8_bm25.py $1 $2
echo "cr_CF_HYB8"

python cr_cf_hyb10_bm25.py $1 $2
echo "cr_CF_HYB10"

python cr_cluster_creative_bm25.py $1 $2
echo "cr_CLUSTER"


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
