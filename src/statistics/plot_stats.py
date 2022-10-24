"""
get all instrument classes which are present in the dataset and plot statistics.
duplicate instruments in each track are not counted.
"""
#TODO: let the user specify if the number of sources or the number of tracks (which contain a specific instrument) should be plotted
#TODO: in barplot --> make bars for train, valid and test set
import yaml
import argparse
import os
import numpy as np
from glob import glob
from collections import Counter
from matplotlib import pyplot as plt
import tqdm

def bar_plot(classes):
    num_classes = len(classes)
    x = np.linspace(0, num_classes, num_classes)
    figsize = (3, 4)
    plt.figure(figsize=figsize)
    y = []
    labels = []
    for row in classes:
        y.append(row[1])
        labels.append(row[0])

    plt.bar(x, y, width=0.6)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('# Sources')
    plt.title(config.plot_title)
    plt.tight_layout()
    filename = f"{config.hierarchical_level}_{config.dataset}_{config.split}.png"
    plt.savefig(os.path.join("../../figs", filename))
    plt.show()


def get_instruments(metadata_files):
    key = "++CLASS++"
    if config.hierarchical_level == "families":
        key = "++FAMILY++"

    all_instruments = []
    has_bleed_count = 0
    for file in tqdm.tqdm(metadata_files):
        with open(file, 'r') as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)

        if config.dataset in ['medleydb', 'mixing-secrets']:
            if doc["has_bleed"] == 'yes':
                has_bleed_count += 1

        inst_one_song = []
        stems = doc["stems"]
        for stem in stems:        
            if config.dataset == 'slakh':
                inst_one_song.append(stems[stem][key])
            else:
                raw_files = stems[stem]["raw"]
                for raw_file in raw_files:
                    inst_one_song.append(raw_files[raw_file][key])
        
        all_instruments += inst_one_song

    return all_instruments, has_bleed_count


def main(config):
    metadata_path = "../../data/splits"
    if config.split == 'all':
        metadata_files = glob(os.path.join(metadata_path, config.dataset, "*", "*.yaml"))
    else:
        metadata_files = glob(os.path.join(metadata_path, config.dataset, config.split, "*.yaml"))

    all_instruments, has_bleed_count = get_instruments(metadata_files)
    all_unique_instruments = set(all_instruments)

    # --- print stats ---
    print(f"number of sources: {len(all_instruments)}")
    print(f"number of tracks: {len(metadata_files)}")
    print(f"number of tracks with crosstalk: {has_bleed_count}")
    print(f"number of classes: {len(all_unique_instruments)}")
    print(all_unique_instruments)

    # plot only class which are defined in classes.yaml
    cleaned_classes = []
    with open("./../training/classes.yaml", 'r') as stream:
        data_dict = yaml.safe_load(stream)
    instr_families = data_dict['instr_families']
    instr_classes = data_dict['instr_classes']

    if config.hierarchical_level == "families":
        for fam in instr_families:
            for inst in all_instruments:
                if inst == fam:
                    cleaned_classes.append(inst)

    if config.hierarchical_level == "classes":
        for cls in instr_classes:
            for inst in all_instruments:
                if inst == cls:
                    cleaned_classes.append(inst)


    classes_with_counts = Counter(cleaned_classes).most_common()

    bar_plot(classes_with_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medleydb', 
                        choices=['slakh', 'medleydb', 'mixing-secrets'])
    parser.add_argument('--split', type=str, default='all', 
                        choices=['train', 'valid', 'test', 'all'])
    parser.add_argument('--hierarchical_level', type=str, default='classes', 
                        choices=['families', 'classes'])
    parser.add_argument('--plot_title', type=str, default='')

    config = parser.parse_args()

    print(config)
    main(config)