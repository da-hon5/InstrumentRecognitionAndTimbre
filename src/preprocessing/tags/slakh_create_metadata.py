"""
copies metadata.yaml files from raw slakh dataset to 'data/metadata/slakh' 
in the repo     !!! run this script first !!!
"""
from glob import glob
import os
import tqdm
import shutil

# choose audio data path
audiodata_path = "/srv/ALL/datasets"
metadata_path = "../../../data/metadata"


def main():
    new_folder = os.path.join(metadata_path, "slakh")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    metadata_files = glob(os.path.join(audiodata_path, "raw", "slakh2100", "*/*/metadata.yaml"))

    for file in tqdm.tqdm(metadata_files):
        track_name = file.split('/')[-2]
        new_path = os.path.join(new_folder, track_name + '.yaml')
        shutil.copy(file, new_path)


if __name__ == "__main__":
    main()