# Multi-Instrument Recognition, Mix-Parameter Estimation and Timbre Characterization using Deep Neural Networks
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

blablabla ...
TODO: describe all the preprocessing steps in detail ... when to run which script

## Reference

**put a link to my master thesis here?**



## Available Models
- **Harmonic CNN** : Data-Driven Harmonic Filters for Audio Representation Learning, Won et al., 2020 [[pdf](https://ccrma.stanford.edu/~urinieto/MARL/publications/ICASSP2020_Won.pdf)]
- **Short-chunk CNN** : Prevalent 3x3 CNN. So-called *vgg*-ish model with a small receptieve field.
- **Short-chunk CNN + Residual** : Short-chunk CNN with residual connections.


## Requirements
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```


## Preprocessing
STFT will be done on-the-fly. You only need to resample audio files and convert them to single channel `.wav` files. 

`cd src/preprocessing/audio`

`python preprocessing.py`

Options

```
'--dataset', type=str, default='mtat', 
    choices=['mtat', 'jamendo', 'medleydb', 'slakh', 'musdb18']
'--samplerate', type=int, default=32000
'--raw_datapath', type=str, default='./data'
'--preprocessed_datapath', type=str, default='./data'
```

## Training

`cd src/training/`

`python train.py -c config.yaml`  
You can override all parameters in config.yaml with the syntax `--classname.parameter`.  
For example: `python train.py -c config.yaml --DataConfig.data_path /your/data/path`  
Visit [spock](https://fidelity.github.io/spock/) for more infos.


## Evaluation
`cd src/training/`

`python eval.py -c config.yaml`