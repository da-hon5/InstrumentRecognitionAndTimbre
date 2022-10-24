"""
check if some audio files from medleydb are broken and delete corresponding metadata file
"""
#TODO: also check STEMS and RAW folders if files are broken

import soundfile
import os
import shutil
from glob import glob

# choose medleydb version ('v1' or 'v2')
version = 'v2'

remove_audio = False   # if True -> audio files are also removed

DATA_PATH = "/clusterFS/home/student/hbradl/data"

def main():
    search_folder = os.path.join(DATA_PATH, "medleydb", version.upper())
    sub_dirs = glob(str(search_folder) + '/*/')

    for dir in sub_dirs:
        songname = dir.split("/")[-2]
        wav_path = dir + songname + "_MIX.wav"
        try:
            npy, _ = soundfile.read(wav_path)
        except RuntimeError:
            print(wav_path)
            remove_metadata_from_dataset(songname)
            if remove_audio:
                remove_audio_from_dataset(dir)



def remove_metadata_from_dataset(songname):
    yaml_file = songname + "_METADATA.yaml"
    yaml_path = os.path.join("./../../../data/metadata/medleydb", version, yaml_file)
    if os.path.exists(yaml_path):
        os.remove(yaml_path)


def remove_audio_from_dataset(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


if __name__ == "__main__":
    main()