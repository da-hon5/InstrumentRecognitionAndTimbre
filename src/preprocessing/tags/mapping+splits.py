"""
hierarchical mapping of labels + creates train/valid/test splits
"""
import os
import argparse
import tqdm
import yaml
import random
import shutil
from glob import glob
import numpy as np

random.seed(0)

data_path = "./../../../data"

heavy_bleed_songs = ['SelwynJazz_MuchTooMuch', 'MR0804_JesperBuhlTrio', 'HowToKillAConversation_Mute',
        'FunnyValentines_SleighRide', 'BigStoneCulture_FragileThoughts',
        'AsamClassicalSoloists_NonLoDiroColLabbro', 'MichaelKropf_AllGoodThings', 'Wolf_DieBekherte', 'Schubert_Erstarrung', 
        'Mozart_DiesBildnis', 'Mozart_BesterJungling', 'Debussy_LenfantProdigue', 
        'Handel_TornamiAVagheggiar', 'Phoenix_BrokenPledgeChicagoReel', 'Phoenix_ElzicsFarewell',
        'Phoenix_ColliersDaughter', 'Phoenix_LarkOnTheStrandDrummondCastle', 
        'JoelHelander_IntheAtticBedroom', 'Phoenix_SeanCaughlinsTheScartaglen', 
        'CroqueMadame_Oil', 'CroqueMadame_Pilot', 'MatthewEntwistle_ReturnToVenezia', 
        'Verdi_IlTrovatore', 'TemperedElan_DorothyJeanne', 'AmadeusRedux_MozartAllegro', 
        'AmadeusRedux_SchubertMovement3', 'TemperedElan_WiseOne', 'PoliceHearingDepartment_Brahms', 
        'TleilaxEnsemble_Late', 'AmadeusRedux_SchubertMovement2', 'TheFranckDuo_FranckViolinSonataInAMajorMovement1', 
        'PoliceHearingDepartment_VivaldiMovement1', 'PoliceHearingDepartment_VivaldiMovement2', 
        'PoliceHearingDepartment_VivaldiMovement3', 'Allegria_MendelssohnMovement1', 'TleilaxEnsemble_MelancholyFlowers', 
        'FallingSparks_PakKlongTalad', 'FallingSparks_Improvisation1', 'FallingSparks_Improvisation2', 
        'FallingSparks_Improvisation3', 'TheFranckDuo_FranckViolinSonataInAMajorMovement2', 
        'DeclareAString_MendelssohnPianoTrio1Movement1', 'Karachacha_Volamos', 'FennelCartwright_FlowerDrumSong', 
        'TheNoHoStringOrchestra_ElgarMovement1', 'TheNoHoStringOrchestra_ElgarMovement2', 
        'TheNoHoStringOrchestra_ElgarMovement3', 'DeclareAString_MendelssohnPianoTrio1Movement2', 
        'FennelCartwright_DearTessie', 'Katzn_CharlieKnox', 'FennelCartwright_WinterLake', 
        'HopsNVinyl_WidowsWalk', 'HopsNVinyl_HoneyBrown', 'HopsNVinyl_ReignCheck', 
        'PoliceHearingDepartment_SchumannMovement1', 'PoliceHearingDepartment_SchumannMovement2', 
        'DahkaBand_SoldierMan', 'SasquatchConnection_Illuminati', 'QuantumChromos_Circuits', 
        'SasquatchConnection_Struttin', 'RodrigoBonelli_BalladForLaura', 'SasquatchConnection_ThanksALatte', 
        'SasquatchConnection_BoomBoxing', 'SasquatchConnection_HolyClamsBatman', 'HopsNVinyl_ChickenFriedSteak', 
        'Schumann_Mignon', 'TablaBreakbeatScience_WhoIsIt']


def get_metadata_files():
    metadata_files = glob(os.path.join(data_path, "metadata", config.dataset, "*.yaml"))
    metadata_files.sort() #to get same result on different machines (order of glob can vary)
    return metadata_files


def remove_heavy_bleeding_songs(metadata_files):
    cleaned_files = []
    for idx, file in enumerate(metadata_files):
        if file.split('/')[-1][:-14] not in heavy_bleed_songs:
            cleaned_files.append(metadata_files[idx])

    return cleaned_files


def getpath(nested_dict, value, prepath=()):
        """ recursive function to search for a value in a nested dict
        and return the path (slightly modified version from stackoverflow) """
        for k, v in nested_dict.items():
            path = prepath + (k,)
            if v is not None:
                if value in v and not isinstance(v, dict): # found value
                    return path
                elif isinstance(v, dict): # v is a dict
                    p = getpath(v, value, path) # recursive call
                    if p is not None:
                        return p


def hierarchical_mapping(orig_label):
    """ takes in original label, outputs hierarchical labels on 3 levels
    level0 --> family
    level1 --> class
    level2 --> subclass """
    instr_fam, instr_class, instr_subclass = None, None, None

    with open("taxonomy.yaml", 'r') as f:
        taxonomy_dict = yaml.load(f, Loader=yaml.FullLoader)

    path = getpath(taxonomy_dict, orig_label)
    #print(path)
    if path is not None:
        instr_fam = path[0]
        instr_class = path[1]
        if len(path) == 3:
            instr_subclass = path[2]

    return instr_fam, instr_class, instr_subclass


def create_hierarchical_labels(file):
    with open(file, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)

    stems = doc["stems"]
    for stem in stems:
        if config.dataset == 'slakh':
            orig_label = stems[stem]["midi_program_name"]
            instr_fam, instr_class, instr_subclass = hierarchical_mapping(orig_label)
            stems[stem]["++FAMILY++"] = instr_fam
            stems[stem]["++CLASS++"] = instr_class
            stems[stem]["++SUBCLASS++"] = instr_subclass
        else:
            raw_files = stems[stem]["raw"]
            for raw_file in raw_files:
                orig_label = raw_files[raw_file]["instrument"]
                instr_fam, instr_class, instr_subclass = hierarchical_mapping(orig_label)
                raw_files[raw_file]["++FAMILY++"] = instr_fam
                raw_files[raw_file]["++CLASS++"] = instr_class
                raw_files[raw_file]["++SUBCLASS++"] = instr_subclass

    return doc


def save_doc_to_yaml(doc, save_path, filename):
    new_metadata_path = os.path.join(save_path, filename)
    with open(new_metadata_path, 'w+') as f:
        yaml.dump(doc, f, indent=2, default_flow_style=False)


def create_folders():
    valid_path = os.path.join(data_path, "splits", config.dataset, "valid")
    test_path = os.path.join(data_path, "splits", config.dataset, "test")
    train_path = os.path.join(data_path, "splits", config.dataset, "train")

    # remove old split
    if os.path.exists(valid_path):
        shutil.rmtree(valid_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    # create valid, test and train directories
    os.makedirs(valid_path)
    os.makedirs(test_path)
    os.makedirs(train_path)

    return train_path, valid_path, test_path


def clean_metadata(metadata_files):
    audio_folders = glob(os.path.join(config.audiodata_path, "mixing-secrets", "*"))
    audio_folders = [folder.split('/')[-1][:-5] for folder in audio_folders]

    for file in metadata_files:
        # remove metadata when no audio available
        fn = file.split('/')[-1][:-14] 
        if fn not in audio_folders:
            print(file)
            metadata_files.remove(file)
        else:
            # remove metadata when stem contains multiple instruments
            with open(file, 'r') as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)

            stems = doc["stems"]
            for stem in stems:
                instrument_class = stems[stem]["instrument"]
                if isinstance(instrument_class, list) and (len(instrument_class) != 1):
                    if os.path.exists(file):
                        print(file)
                        metadata_files.remove(file)

    return metadata_files


def main():
     # remove songs with heavy bleed
    remove_bleed = True

    # choose split percentages in %
    valid_percent = 0.175
    test_percent = 0.175
    train_percent = 0.65

    if valid_percent + test_percent + train_percent != 1:
        raise ValueError("train/valid/test percentages have to sum up to 1")

    metadata_files = get_metadata_files()

    if config.dataset == 'mixing-secrets':
        metadata_files = clean_metadata(metadata_files)

    if remove_bleed:
        metadata_files = remove_heavy_bleeding_songs(metadata_files)

    num_files = len(metadata_files)
    num_valid = np.floor(valid_percent * num_files)
    num_test = np.floor(test_percent * num_files)
    num_train = num_files - num_valid - num_test

    train_path, valid_path, test_path = create_folders()

    # make split -> save metadata files in train/valid/test folders
    train_count = 0
    valid_count = 0
    test_count = 0
    for _ in tqdm.tqdm(range(num_files)):
        file = random.choice(metadata_files)
        metadata_files.remove(file)
        doc = create_hierarchical_labels(file)
        fn = file.split('/')[-1]
        if valid_count < num_valid:
            save_doc_to_yaml(doc, save_path=valid_path, filename=fn)
            valid_count += 1
            continue
        if test_count < num_test:
            save_doc_to_yaml(doc, save_path=test_path, filename=fn)
            test_count += 1
            continue
        if train_count < num_train:
            save_doc_to_yaml(doc, save_path=train_path, filename=fn)
            train_count += 1
            continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medleydb', 
                    choices=['medleydb', 'slakh', 'mixing-secrets'])
    parser.add_argument('--audiodata_path', type=str, default='')

    config = parser.parse_args()

    print(config)
    main()