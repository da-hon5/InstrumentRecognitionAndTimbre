"""
takes all audio files from the specified dataset, resamples them to the target sample rate
and converts them to single-channel wav-files
"""
import os
import argparse
import glob
import soundfile
import tqdm
import librosa
import torchaudio
import math

def get_file_paths():
    # get audio file paths
    if config.dataset in ['mtat', 'jamendo']:
        files = glob.glob(os.path.join(config.raw_datapath, config.dataset, 'mp3', '*/*.mp3'))
    elif config.dataset == 'medleydb':
        mixes = glob.glob(os.path.join(config.raw_datapath, config.dataset, '*', '*/*.wav'))
        stems_and_raw = glob.glob(os.path.join(config.raw_datapath, config.dataset, '*', '*/*/*.wav'))
        files = mixes + stems_and_raw
    elif config.dataset == 'slakh':
        mixes = glob.glob(os.path.join(config.raw_datapath, config.dataset+"2100", '*/*/*.flac'))
        stems = glob.glob(os.path.join(config.raw_datapath, config.dataset+"2100", '*/*/*/*.flac'))
        files = mixes + stems
    elif config.dataset == 'musdb18':
        files = glob.glob(os.path.join(config.raw_datapath, config.dataset+"-hq", '*/*/*.wav'))
    elif config.dataset == 'mixing-secrets':
        files = glob.glob(os.path.join(config.raw_datapath, config.dataset, '*/*.wav'))

    return files


def create_dir():
    # create new "wav" directory
    wav_dir = os.path.join(config.preprocessed_datapath, config.dataset, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    return wav_dir


def create_medleydb_folders(folder_path):
    # create medleydb folder structure
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    stem_dir = os.path.join(folder_path, "STEMS")
    if not os.path.exists(stem_dir):
        os.makedirs(stem_dir)
    raw_dir = os.path.join(folder_path, "RAW")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)


def get_longest_source(files):
    """returns dict with fouldername as key and duration (in sec) of longest source as value"""
    dur_dict = {}
    for file in tqdm.tqdm(files):
        foldername = file.split('/')[-2]
        si, _ = torchaudio.info(file)
        source_dur = (si.length / si.channels) / si.rate
        if foldername not in dur_dict.keys():
            dur_dict[foldername] = source_dur
        else:
            if source_dur > dur_dict[foldername]:
                dur_dict[foldername] = source_dur

    return dur_dict



def main(config):

    files = get_file_paths()

    wav_dir = create_dir()

    if config.dataset == 'mixing-secrets':
        duration_dict = get_longest_source(files)

    for fn in tqdm.tqdm(files):
        new_fn = fn.split('/')[-1].split('.')[0] + ".wav"

        if config.dataset == 'medleydb':
            if '_MIX' in new_fn:
                foldername = '_'.join(new_fn.split('_')[:-1])
                new_path = os.path.join(wav_dir, foldername, new_fn)
            elif '_STEM_' in new_fn:
                foldername = '_'.join(new_fn.split('_')[:-2])
                new_path = os.path.join(wav_dir, foldername, "STEMS", new_fn)
            elif '_RAW_' in new_fn:
                foldername = '_'.join(new_fn.split('_')[:-3])
                new_path = os.path.join(wav_dir, foldername, "RAW", new_fn)

            folder_path = os.path.join(wav_dir, foldername)
            create_medleydb_folders(folder_path)

        elif config.dataset in ['mtat', 'jamendo']:
            new_path = os.path.join(wav_dir, new_fn)

        elif config.dataset == 'slakh':
            if new_fn == 'mix.wav':
                foldername = fn.split('/')[-2]
                new_path = os.path.join(wav_dir, foldername, new_fn)
            elif new_fn[0] == 'S':
                foldername = fn.split('/')[-3]
                new_path = os.path.join(wav_dir, foldername, 'stems', new_fn)
            folder_path = '/'.join(new_path.split('/')[:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        elif config.dataset == 'musdb18':
            foldername = fn.split('/')[-2]
            split = fn.split('/')[-3]
            new_path = os.path.join(wav_dir, split, foldername, new_fn)
            folder_path = '/'.join(new_path.split('/')[:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        elif config.dataset == 'mixing-secrets':
            foldername = fn.split('/')[-2]
            new_path = os.path.join(wav_dir, foldername, new_fn)
            folder_path = '/'.join(new_path.split('/')[:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


        if not os.path.exists(new_path):
            try:
                audio, _ = librosa.core.load(fn, sr=config.samplerate, mono=True, res_type='kaiser_fast')
                if config.dataset == 'mixing-secrets':
                    max_duration = math.ceil(duration_dict[foldername] * config.samplerate)
                    audio = librosa.util.fix_length(audio, max_duration)
                soundfile.write(new_path, audio, config.samplerate, 'PCM_16')
            except RuntimeError:
                # some audio files are broken
                print(f'broken file: {fn}')
                continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='mtat', 
                    choices=['mtat', 'jamendo', 'medleydb', 'slakh', 'musdb18', 'mixing-secrets'])
    parser.add_argument('--samplerate', type=int, default=32000)
    parser.add_argument('--raw_datapath', type=str, default='./data')
    parser.add_argument('--preprocessed_datapath', type=str, default='./data')

    config = parser.parse_args()

    print(config)
    main(config)
