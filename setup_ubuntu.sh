#!/bin/bash

echo ""
echo ">>>>>>>>>>>>>>>>>> setup python <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""

sudo apt-get update
sudo apt-get install python-pip
sudo apt-get install python3
sudo apt-get install gcc
sudo apt-get install libxml2-dev libxslt1-dev
sudo apt-get install python-dateutil python-docutils python-feedparser python-gdata python-jinja2 python-ldap
sudo apt-get install python-libxslt1 python-lxml python-mako python-mock python-openid python-psycopg2 python-psutil python-pybabel
sudo apt-get install python-pychart python-pydot python-pyparsing python-reportlab python-simplejson python-tz python-unittest2
sudo apt-get install python-vatnumber python-vobject python-webdav python-werkzeug python-xlwt python-yaml python-zsi
sudo apt-get install python3-dev
sudo apt-get install unzip
sudo apt-get install python3-tk
sudo apt install virtualenv

virtualenv -p /usr/bin/python3 py3env

sleep 1
source py3env/bin/activate

echo ""
echo ">>>>>>>>>>>>>>>>>> installing python packages <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""
sleep 1
pip install pandas scipy numpy cython matplotlib tqdm sklearn psutil nltk deap regex
pip install bayesian-optimization spotipy


echo ""
echo ">>>>>>>>>>>>>>>>>> installing FireFlies REPO  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""
sleep 1
python setup.py sdist
pip install -e .

echo ""
echo ">>>>>>>>>>>>>>>>>> compiling cython simone's module  <<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""
sleep 1
cd recommenders/similarity
python compileCython.py build_ext --inplace
cd ../..


echo ""
echo ">>>>>>>>>>>>>>>>>> end  <<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""



