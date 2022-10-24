"""
helper functions to be used in other python files
"""
import os
from glob import glob
import yaml
import numpy as np
import soundfile
import torchaudio
from sklearn import metrics

import config


def db_to_lin(gain_db):
    return 10 ** (gain_db / 20)


def get_songlength(wav_path):
    return torchaudio.info(wav_path)[0].length


def peak_normalize(audio):
    return audio / np.max(np.abs(audio))


def peak_normalize_if_clipping(audio):
    """normalize audio when max amplitude bigger than 1;
    returns tuple (normalized_audio, is_clipping)"""
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < 1:
        normalized_audio = audio
        is_clipping = False
    else:
        normalized_audio = audio / np.max(np.abs(audio))
        is_clipping = True
    return normalized_audio, is_clipping


def get_auc(est_array, gt_array):
    # macro -> calculate metrics for each label and find their mean
    roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    return roc_aucs, pr_aucs


def get_accuracy(est_array, gt_array):
    """returns accuracy for each class"""
    accuracies = []
    for i in range(np.shape(est_array)[1]):
        y_pred = est_array[:, i].astype(int)
        y_true = gt_array[:, i].astype(int)
        acc = metrics.accuracy_score(y_true, y_pred)
        accuracies.append(acc)
    return accuracies


def get_precision(est_array, gt_array):
    """returns precision for each class"""
    #NOTE: When true positive + false positive == 0, precision returns 0 and raises UndefinedMetricWarning
    precisions = []
    for i in range(np.shape(est_array)[1]):
        y_pred = est_array[:, i].astype(int)
        y_true = gt_array[:, i].astype(int)
        prec = metrics.precision_score(y_true, y_pred)
        precisions.append(prec)
    return precisions


def get_recall(est_array, gt_array):
    """returns recall for each class"""
    #NOTE: When true positive + false negative == 0, recall returns 0 and raises UndefinedMetricWarning
    recalls = []
    for i in range(np.shape(est_array)[1]):
        y_pred = est_array[:, i].astype(int)
        y_true = gt_array[:, i].astype(int)
        rec = metrics.recall_score(y_true, y_pred)
        recalls.append(rec)
    return recalls


def get_f1_score(est_array, gt_array):
    """returns f1-score for each class"""
    #NOTE: When either precision or recall is 0, f1-score returns 0 and raises UndefinedMetricWarning
    f1_scores = []
    for i in range(np.shape(est_array)[1]):
        y_pred = est_array[:, i].astype(int)
        y_true = gt_array[:, i].astype(int)
        f1 = metrics.f1_score(y_true, y_pred)
        f1_scores.append(f1)
    return f1_scores


def write_audio_to_file(audio, source, loudn, tags):
    """just for debugging"""
    soundfile.write(f"{source}_{loudn:.2f}_{tags}.wav", audio, config.samplerate)


def delete_wavs(relative_path=None):
    """just for debugging"""
    current_dir = os.getcwd()
    if relative_path is None:
        dir = current_dir
    else:
        dir = os.path.join(current_dir, relative_path)
    files = glob(os.path.join(dir, "*.wav"))
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def check_if_all_classes_exist(gt_array, classes):
        """
        checks if there is at least 1 sample per class (instrument) in the valid/test set 
        and if no class is present in all samples! -> otherwise roc-auc can not be computed!
        """
        mean_per_class = np.mean(gt_array, axis=0)
        not_represented = []
        overrepresented = []
        for idx, clss in enumerate(mean_per_class):
            if clss == 0:
                not_represented.append(classes[idx])
            if clss == 1:
                overrepresented.append(classes[idx])

        error_msg = ""  
        if len(not_represented) > 0:
            error_msg += f"class(es) {not_represented} not represented in the data!\n"
        if len(overrepresented) > 0:
            error_msg += f"class(es) {overrepresented} present in all samples in the data!\n"
        if len(error_msg) > 0:
            #raise RuntimeError(error_msg)
            print(error_msg)
            return False
        else:
            return True


def load_classes(type='instr_families'):
    with open("classes.yaml", 'r') as stream:
        data_dict = yaml.safe_load(stream)
    return data_dict[type]


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f)
    return data


def squeeze_if_tensor_is_3dimensional(tensor, dim=0):
    if tensor.dim() == 3:
        return tensor.squeeze(dim)
    else:
        return tensor


def iqr(x, axis=1):
    """compute interquartile range over specified axis"""
    q75, q25 = np.percentile(x, [75 ,25], axis=axis)
    return q75 - q25
