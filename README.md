# Deep Neural Networks for Multi-Instrument Recognition and Timbre Characterization

This repo was created in the course of my master's thesis at Graz University of Technology in 2022.


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
