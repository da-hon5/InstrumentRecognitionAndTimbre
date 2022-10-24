#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TrainConfig.mode from-scratch --TrainConfig.rand_seed 622 --TrainConfig.model_save_path ./../../models/transfer-learning-method-experiment/seed=622/from-scratch --TrainConfig.backbone_lr 1e-2
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TrainConfig.mode frozen-backbone --TrainConfig.rand_seed 622 --TrainConfig.model_save_path ./../../models/transfer-learning-method-experiment/seed=622/frozen-backbone
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --TrainConfig.mode finetuning --TrainConfig.rand_seed 622 --TrainConfig.model_save_path ./../../models/transfer-learning-method-experiment/seed=622/finetuning
