#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "p_multiple_songs = 0.0"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=0.0/best_model.pth
echo "--------------------------------------"
echo "p_multiple_songs = 0.2"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=0.2/best_model.pth
echo "--------------------------------------"
echo "p_multiple_songs = 0.4"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=0.4/best_model.pth
echo "--------------------------------------"
echo "p_multiple_songs = 0.6"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=0.6/best_model.pth
echo "--------------------------------------"
echo "p_multiple_songs = 0.8"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=0.8/best_model.pth
echo "--------------------------------------"
echo "p_multiple_songs = 1.0"
python test.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TestConfig.model_load_path ./../../models/p_mult_songs-experiment/seed=422/p_mult_songs=1.0/best_model.pth