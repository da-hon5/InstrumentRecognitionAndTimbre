"""
creates audio chunks to use for t-sne visualization
"""
import os
import yaml
from glob import glob
from sklearn.manifold import TSNE
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile
import librosa
import tqdm
import torchaudio
import pyloudnorm as pyln
import essentia.standard as ess
import soundfile
import plotly.express as px
from torch.utils.tensorboard import SummaryWriter


audiodata_path = '/srv/ALL/datasets/preprocessed'
audio_save_path = '/srv/ALL/datasets/preprocessed/t-sne_chunks'
samplerate = 32000


def get_song_info(doc, dataset):
    source_filenames = []
    instr_fams = []
    instr_classes = []
    instr_subclasses = []
    stems = doc["stems"]

    if dataset == 'slakh':
        song_name = doc["audio_dir"].split('/')[0]
        for stem in stems:
            if stems[stem]["audio_rendered"]:
                source_filenames.append(stem + ".wav")
                instr_fams.append(stems[stem]["++FAMILY++"])
                instr_classes.append(stems[stem]["++CLASS++"])
                instr_subclasses.append(stems[stem]["++SUBCLASS++"])
    else:
        song_name = doc["mix_filename"][0:-8]
        for stem in stems:
            raw_files = stems[stem]["raw"]

            for raw_file in raw_files:
                source_filenames.append(raw_files[raw_file]["filename"])
                instr_fams.append(raw_files[raw_file]["++FAMILY++"])
                instr_classes.append(raw_files[raw_file]["++CLASS++"])
                instr_subclasses.append(raw_files[raw_file]["++SUBCLASS++"])

    return song_name, source_filenames, instr_fams, instr_classes, instr_subclasses


def get_audio_chunk(wav_path):
    ''' returns chunk with the highest rms! 
    if all chunks are below a threshold --> returns None '''
    rms_threshold = 10 ** (-60 / 20)
    chunk_length = 4 * samplerate

    song_length = torchaudio.info(wav_path)[0].length
    hop = (song_length - chunk_length) // 8

    chunks = []
    rms = []
    for i in range(8):
        audio, _ = soundfile.read(wav_path, start=i*hop, frames=chunk_length, dtype='float32')
        chunks.append(audio)
        rms.append(np.sqrt(np.mean(np.square(audio))))

    max_rms = max(rms)
    if max_rms < rms_threshold:
        return None

    max_index = rms.index(max_rms)
    return chunks[max_index]


def get_metadata_files(dataset):
    return glob(os.path.join("./../../data/splits", dataset, "*", "*.yaml"))


def get_wav_path(dataset, song_name, source):
    if dataset == 'slakh':
        wav_path = os.path.join(audiodata_path, 'slakh', "wav", song_name, "stems", source)
    elif dataset == 'mixing-secrets':
        wav_path = os.path.join(audiodata_path, 'mixing-secrets', "wav", song_name + '_Full', source)
    elif dataset == 'medleydb':
        wav_path = os.path.join(audiodata_path, 'medleydb', "wav", song_name, "RAW", source)
    return wav_path


def peak_normalize(audio):
    return audio / np.max(np.abs(audio))


def save_wav(audio, song_name, source_filename, instr_fam, instr_class):
    dir = os.path.join(audio_save_path, instr_fam, instr_class)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, song_name+source_filename)
    soundfile.write(path, audio, samplerate=samplerate)


def make_audio_chunks(datasets=[]):
    for dataset in datasets:
        metadata_files = get_metadata_files(dataset)
        for file in tqdm.tqdm(metadata_files):
            with open(file, 'r') as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)

            song_name, source_filenames, instr_fams, instr_classes, instr_subclasses = get_song_info(doc, dataset)
            for i in range(len(source_filenames)):
                wav_path = get_wav_path(dataset, song_name, source=source_filenames[i])
                audio = get_audio_chunk(wav_path)
                if audio is not None and instr_fams[i] is not None:
                    audio = peak_normalize(audio)
                    save_wav(audio, song_name, source_filenames[i], instr_fams[i], instr_classes[i])




def main():
    make_audio_chunks(datasets=['medleydb', 'mixing-secrets'])



if __name__ == '__main__':
    main()