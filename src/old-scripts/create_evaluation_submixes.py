"""
create submixes for validation and test set for the multitrack datasets
"""
from glob import glob
import yaml
import os
import soundfile
import numpy as np
import tqdm
import random
import argparse
from shutil import copyfile
from audiomentations import Compose, TimeStretch, PitchShift


def make_dirs():
    folder_path = os.path.join(config.audiodata_path, config.dataset, config.split + "_submixes")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_old_files():
    files = glob(os.path.join(config.audiodata_path, config.dataset, config.split + "_submixes", "*.wav"))
    for file in files:
        os.remove(file)


def get_metadata_files():
    if config.dataset in ['medleydb', 'mixing-secrets']:
        metadata_path = "../../../data/splits"
    elif config.dataset == 'slakh':
        metadata_path = "../../../data/metadata/preprocessed"

    metadata_files = glob(os.path.join(metadata_path, config.dataset, config.split, "*.yaml"))
    metadata_files.sort() #to get same result on different machines (order of glob can vary)
    return metadata_files


def create_random_submixes(stems):
    num_sources = len(stems)
    available_stems = list(stems)
    stems_for_submixes = {0: [], 1: [], 2:[]}
    if num_sources >= 9:
        stems_for_submixes[0] = random.sample(available_stems, num_sources // 3)
        available_stems = [stem for stem in available_stems if stem not in stems_for_submixes[0]]
        stems_for_submixes[1] = random.sample(available_stems, num_sources // 3)
        available_stems = [stem for stem in available_stems if stem not in stems_for_submixes[1]]
        stems_for_submixes[2] = available_stems
    elif num_sources < 9 and num_sources >= 4:
        stems_for_submixes[0] = random.sample(available_stems, num_sources // 2)
        available_stems = [stem for stem in available_stems if stem not in stems_for_submixes[0]]
        stems_for_submixes[1] = available_stems

    submixes = {0: {'files': [], 'tags': []},
                1: {'files': [], 'tags': []},
                2: {'files': [], 'tags': []}}

    for stem in stems:
        if config.dataset in ['medleydb', 'mixing-secrets']:
            filename = stems[stem]["filename"]
        elif config.dataset == 'slakh':
            filename = stem + '.wav'
            if not stems[stem]["audio_rendered"]:
                continue
        instr_fam = stems[stem]["instr-family"]
        instr_class = stems[stem]["instr-class"]

        for i in range(3):
            if stem in stems_for_submixes[i]:
                if config.dataset == 'mixing-secrets' and '_STEM_' in filename:
                    raw_files = stems[stem]["raw"]
                    for raw_file in raw_files:
                        submixes[i]['files'].append(raw_files[raw_file]['filename'])
                else:
                    submixes[i]['files'].append(filename)
                if len(instr_fam) > 0:
                    submixes[i]['tags'].append(instr_fam)
                    if len(instr_class) > 0:
                        submixes[i]['tags'].append(instr_class)

    return submixes


def create_submix_files(song_folder, submixes):
    for i, submix in submixes.items():
        audio_mix = np.zeros(1, dtype='float32')
        for file in submix['files']:
            if config.dataset == "medleydb":
                wav_path = os.path.join(config.audiodata_path, config.dataset, "wav", song_folder, "STEMS", file)
            elif config.dataset == 'mixing-secrets':
                wav_path = os.path.join(config.audiodata_path, config.dataset, "wav", song_folder, file)
            elif config.dataset == 'slakh':
                wav_path = os.path.join(config.audiodata_path, config.dataset, "wav", song_folder, "stems", file)

            audio, sr = soundfile.read(wav_path, dtype='float32')
            audio_mix = np.add(audio_mix, audio)

        tags = '&'.join(list(set(submix['tags'])))
        if len(tags) > 0:
            if config.dataset == 'medleydb':
                new_filename = f"{file[:-12]}_submix{i+1}#{tags}#.wav"
            elif config.dataset in ['mixing-secrets', 'slakh']:
                new_filename = f"{song_folder}_submix{i+1}#{tags}#.wav"
            new_path = os.path.join(config.audiodata_path, config.dataset, config.split + "_submixes", new_filename)

            #augmentation
            augment = Compose([TimeStretch(min_rate=0.9, max_rate=1.2, p=0.5),
                                PitchShift(min_semitones=-2, max_semitones=2, p=0.8)],
                                p=1.0, shuffle=True)
            audio_mix = augment(audio_mix, sample_rate=sr)

            #peak normalize audio
            audio_mix = audio_mix / np.max(np.abs(audio_mix))

            soundfile.write(new_path, audio_mix, sr, 'PCM_16')


def create_mixing_secrets_mix(stems, song_folder):
    mix = {'files': [], 'tags': []}
    for stem in stems:
        filename = stems[stem]["filename"]
        instr_fam = stems[stem]["instr-family"]
        instr_class = stems[stem]["instr-class"]

        if '_STEM_' in filename:
            raw_files = stems[stem]["raw"]
            for raw_file in raw_files:
                mix['files'].append(raw_files[raw_file]['filename'])
        else:
            mix['files'].append(filename)
        if len(instr_fam) > 0:
            mix['tags'].append(instr_fam)
            if len(instr_class) > 0:
                mix['tags'].append(instr_class)

    audio_mix = np.zeros(1, dtype='float32')
    for file in mix['files']:
        wav_path = os.path.join(config.audiodata_path, 'mixing-secrets', "wav", song_folder, file)
        audio, sr = soundfile.read(wav_path, dtype='float32')
        audio_mix = np.add(audio_mix, audio)

    #peak normalize audio
    audio_mix = audio_mix / np.max(np.abs(audio_mix))

    tags = '&'.join(list(set(mix['tags'])))
    if len(tags) > 0:
        new_filename = f"{song_folder}_MIX#{tags}#.wav"
        new_mix_path = os.path.join(config.audiodata_path, config.dataset, config.split + "_submixes", new_filename)
        soundfile.write(new_mix_path, audio_mix, sr, 'PCM_16')

def main():
    make_dirs()
    delete_old_files()

    random.seed(config.split)

    metadata_files = get_metadata_files()

    for file in tqdm.tqdm(metadata_files):
        with open(file, 'r') as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)

        stems = doc["stems"]
                
        if config.dataset == 'medleydb':
            song_folder = doc["mix_filename"][:-8]
        elif config.dataset == 'mixing-secrets':
            song_folder = doc["mix_filename"][:-8] + '_Full'
        elif config.dataset == 'slakh':
            song_folder = file.split('/')[-1][:-5]
           
        submixes = create_random_submixes(stems)
        create_submix_files(song_folder, submixes)

        if config.dataset == 'mixing-secrets':
            create_mixing_secrets_mix(stems, song_folder)

        #copy mixes to submix folders
        if config.dataset in ['medleydb', 'slakh']:
            if config.dataset == 'medleydb':
                mix_path = os.path.join(config.audiodata_path, config.dataset, "wav", song_folder, song_folder + '_MIX.wav')
            elif config.dataset == 'slakh':
                mix_path = os.path.join(config.audiodata_path, config.dataset, "wav", song_folder, 'mix.wav')
                
            tags = []
            for stem in stems:
                instr_fam = stems[stem]["instr-family"]
                instr_class = stems[stem]["instr-class"]
                if len(instr_fam) > 0:
                    tags.append(instr_fam)
                    if len(instr_class) > 0:
                        tags.append(instr_class)
            
            tags = '&'.join(list(set(tags)))
            if len(tags) > 0:
                new_filename = f"{song_folder}_MIX#{tags}#.wav"
                new_mix_path = os.path.join(config.audiodata_path, config.dataset, config.split + "_submixes", new_filename)
                copyfile(mix_path, new_mix_path)

          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='slakh', 
                        choices=['slakh', 'medleydb', 'mixing-secrets'])
    parser.add_argument('--split', type=str, default='valid', 
                        choices=['valid', 'test'])
    parser.add_argument('--audiodata_path', type=str, default='')

    config = parser.parse_args()

    print(config)
    main()
