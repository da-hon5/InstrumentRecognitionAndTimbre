import soundfile
import os
import shutil
from glob import glob
import tqdm


DATA_PATH = "/clusterFS/home/student/hbradl/data/preprocessed"

def main():
    audio_files = glob(os.path.join(DATA_PATH, "mixing-secrets", "wav", "*/*.wav"))

    for file in tqdm.tqdm(audio_files):
        #songname = dir.split("/")[-2]
        #wav_path = dir + songname + "_MIX.wav"
        try:
            audio, sr = soundfile.read(file)
        except RuntimeError:
            print(file)
        if len(audio) < 10 * 32000:
            print(file)




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