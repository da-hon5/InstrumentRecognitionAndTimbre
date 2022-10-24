#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

SAVE_PATH="./../../models/feature-prediction/transfer-learning-method-experiment-percussion"


# echo "from-scratch"
# python train.py -c config.yaml --DataConfig.target_class percussion --TrainConfig.mode from-scratch --TrainConfig.backbone_lr 1e-2 --TrainConfig.head_lr 1e-2 --TrainConfig.model_save_path $SAVE_PATH/from-scratch

echo "frozen-backbone"
python train.py -c config.yaml --DataConfig.target_class percussion --TrainConfig.mode frozen-backbone --TrainConfig.backbone_lr 1e-4 --TrainConfig.head_lr 1e-2 --TrainConfig.model_save_path $SAVE_PATH/frozen-backbone

echo "finetuning"
python train.py -c config.yaml --DataConfig.target_class percussion --TrainConfig.mode finetuning --TrainConfig.backbone_lr 1e-4 --TrainConfig.head_lr 1e-2 --TrainConfig.model_save_path $SAVE_PATH/finetuning
