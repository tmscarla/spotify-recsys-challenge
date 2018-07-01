#!/bin/bash


echo ""
echo ">>>>>>>>>>>>>>>>> downloading data folder <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo ""
sleep 1
curl https://void.cat/28146b8981e6ae6caae9b4b42cc5ff4144f354cf -o data/data.zip
unzip -o data/data.zip -d ./