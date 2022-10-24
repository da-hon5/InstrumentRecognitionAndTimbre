#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# compute mean and variance for all classes! in config.yaml --> mode = compute-stats, n_epochs = 1 (or more?), valid samples per epoch = 0 (just use train set)

# families
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class voice --TrainConfig.model_save_path ./../../models/feature-prediction/stats/voice
# python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class percussion --TrainConfig.model_save_path ./../../models/feature-prediction/stats/percussion
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class plucked-str --TrainConfig.model_save_path ./../../models/feature-prediction/stats/plucked-str
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class bowed-str --TrainConfig.model_save_path ./../../models/feature-prediction/stats/bowed-str
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class woodwind --TrainConfig.model_save_path ./../../models/feature-prediction/stats/woodwind
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class brass --TrainConfig.model_save_path ./../../models/feature-prediction/stats/brass
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class key --TrainConfig.model_save_path ./../../models/feature-prediction/stats/key
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class synth --TrainConfig.model_save_path ./../../models/feature-prediction/stats/synth

# classes
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class singer --TrainConfig.model_save_path ./../../models/feature-prediction/stats/singer
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class drums --TrainConfig.model_save_path ./../../models/feature-prediction/stats/drums
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class violin --TrainConfig.model_save_path ./../../models/feature-prediction/stats/violin
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/stats/e-guitar
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class a-guitar --TrainConfig.model_save_path ./../../models/feature-prediction/stats/a-guitar
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class e-bass --TrainConfig.model_save_path ./../../models/feature-prediction/stats/e-bass
python train.py -c config.yaml --DataConfig.data_path /srv/ALL/datasets/preprocessed --DataConfig.target_class piano --TrainConfig.model_save_path ./../../models/feature-prediction/stats/piano