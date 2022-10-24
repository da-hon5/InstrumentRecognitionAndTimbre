#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 0.0 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=0.0
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 0.2 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=0.2
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 0.4 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=0.4
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 0.6 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=0.6
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 0.8 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=0.8
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.p_multiple_songs 1.0 --TrainConfig.rand_seed 722 --TrainConfig.model_save_path ./../../models/p_mult_songs-experiment/seed=722/p_mult_songs=1.0