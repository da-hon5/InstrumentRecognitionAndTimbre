"""
definition of the spock classes
"""

from spock import spock
from spock import SavePath
from typing import List


@spock
class DataConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    # config file values can be overridden via cmd line args: 
    # e.g. python train.py --config config.yaml --DataConfig.samplerate 48000

    # you can also use default values: e.g. model_input_length: int = 4
    data_path: str
    dataset: str
    samplerate: int
    model_input_length: int
    max_num_sources: int    # max number of sources when using multiple songs
    p_single_source: float
    p_multiple_songs: float  # probability that raw files from different songs are mixed together
    p_skip_percussion: float
    p_skip_plucked_str: float
    p_medleydb: float   # probability that train sample comes from medleydb when dataset is 'combined'
    p_mixingsecrets: float  # probability that train sample comes from mixing-secrets when dataset is 'combined'
    p_slakh: float    # probability that train sample comes from slakh when dataset is 'combined'
    loudn_threshold: int
    n_mfccs: int
    predict_loudness: bool
    target_class: str   # only used for timbre-predictor


@spock
class ModelConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    model_type: str
    n_channels: int
    n_fft: int
    n_mels: int


@spock
class TrainConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    #task: str
    mode: str
    model_save_path: SavePath
    model_load_path: str
    rand_seed: int
    n_epochs: int
    batch_size: int
    backbone_lr: float    # when pretraining backbone_lr and head_lr should be equal (1e-4)
    head_lr: float
    lr_scheduler_stepsize: int
    lr_scheduler_gamma: float
    weight_decay: float
    use_optim_schedule: bool
    num_workers: int
    log_step: int
    samples_per_epoch: int


@spock
class ValidConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    num_workers: int
    chunks_per_track: int
    samples_per_epoch: int


@spock
class TestConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    #task: str   # classification / rep-learning
    model_load_path: str
    rand_seed: int
    num_workers: int
    chunks_per_track: int
    samples_per_epoch: int
    use_tensorboard: bool
    p_tensorboard: float  #probability that a sample in testset is logged to tb

@spock
class AugmentationConfig:
    """Basic spock configuration for example purposes

    Attributes:
        dataset: blablabla
        fancy_parameter: parameter that multiplies a value
        fancier_parameter: parameter that gets added to product of val and fancy_parameter
        most_fancy_parameter: values to apply basic algebra to

    """
    p_augment: float
    p_imp_res: float
    p_time_stretch: float
    p_pitch_shift: float
    p_rand_gain: float
    p_peaking_filter: float
    p_lowshelf_filter: float
    p_highshelf_filter: float
