#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

FOLDER="mae-loss"

# ------ train feature predictors ------
# families
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class voice --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/voice
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class percussion --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/percussion
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class plucked-str --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/plucked-str
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class bowed-str --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/bowed-str
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class woodwind --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/woodwind
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class brass --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/brass
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class key --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/key
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class synth --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/synth

# classes
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class singer --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/singer
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class drums --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/drums
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class violin --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/violin
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/e-guitar
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class a-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/a-guitar
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-bass --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/e-bass
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class piano --TrainConfig.model_save_path ./../../models/feature-prediction/$FOLDER/piano