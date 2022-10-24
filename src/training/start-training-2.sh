#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

# ------ train feature predictors ------
# families
#python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class voice --TrainConfig.model_save_path ./../../models/feature-prediction/voice
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class percussion --TrainConfig.model_save_path ./../../models/feature-prediction/percussion
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class plucked-str --TrainConfig.model_save_path ./../../models/feature-prediction/plucked-str
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class bowed-str --TrainConfig.model_save_path ./../../models/feature-prediction/bowed-str
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class woodwind --TrainConfig.model_save_path ./../../models/feature-prediction/woodwind
#python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class brass --TrainConfig.model_save_path ./../../models/feature-prediction/brass
#python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class key --TrainConfig.model_save_path ./../../models/feature-prediction/key
#python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class synth --TrainConfig.model_save_path ./../../models/feature-prediction/synth

# classes
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class singer --TrainConfig.model_save_path ./../../models/feature-prediction/singer
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class drums --TrainConfig.model_save_path ./../../models/feature-prediction/drums
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class violin --TrainConfig.model_save_path ./../../models/feature-prediction/violin
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/e-guitar
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class a-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/a-guitar
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-bass --TrainConfig.model_save_path ./../../models/feature-prediction/e-bass
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class piano --TrainConfig.model_save_path ./../../models/feature-prediction/piano