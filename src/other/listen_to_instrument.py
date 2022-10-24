"""
returns the first stem of a query instrument it can find. 
this script is useful for class mapping to decide in which class to put a certain insrument
"""
import os
from glob import glob
import yaml
import tqdm
from shutil import copyfile

#choose dataset ["slakh", "medleydb", "mixing-secrets"]
dataset = "medleydb"

#choose instrument to search for
query_instr = "bass drum"

# max number of stems to return
max_numb_stems = 20

metadata_path = "../../data/metadata"
audiodata_path = "/clusterFS/home/student/hbradl/data/preprocessed" #"/srv/ALL/datasets/preprocessed"

def remove_out_wavs():
    current_dir = os.getcwd()
    files = glob(os.path.join(current_dir, "out_*_.wav"))
    for file in files:
        os.remove(file)


def main():
    remove_out_wavs()

    if dataset == "slakh":
        metadata_files = glob(os.path.join(metadata_path, "raw", "slakh", "*/*.yaml"))
    elif dataset in ["medleydb", "mixing-secrets"]:
        metadata_files = glob(os.path.join(metadata_path, "raw", dataset, "*.yaml"))

    numb_found_stems = 0
    for file in tqdm.tqdm(metadata_files):
        if numb_found_stems == max_numb_stems:
            break
        with open(file, 'r') as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
            stems = doc["stems"]
            for stem in stems:
                if dataset == 'slakh':
                    instr = stems[stem]["midi_program_name"]
                    if instr == query_instr:
                        if stems[stem]["audio_rendered"]:
                            track = file.split('/')[-1][:-5]
                            wav_path = os.path.join(audiodata_path, "slakh", "wav", track, "stems", stem + ".wav")
                            copyfile(wav_path, f"out_{numb_found_stems}_.wav")
                            numb_found_stems += 1
                        else:
                            continue

                elif dataset == 'medleydb':
                    raw_files = stems[stem]["raw"]
                    for raw_file in raw_files:
                        instr = raw_files[raw_file]["instrument"]
                        if instr == query_instr:
                            track = file.split('/')[-1][:-14]
                            filename = f'{track}_RAW_{stem[1:]}_{raw_file[1:]}.wav'
                            wav_path = os.path.join(audiodata_path, "medleydb", "wav", track, "RAW", filename)
                            copyfile(wav_path, f"out_{numb_found_stems}_.wav")
                            numb_found_stems += 1

                elif dataset == 'mixing-secrets':
                    #TODO
                    raw_files = stems[stem]["raw"]
                    for raw_file in raw_files:
                        instr = raw_files[raw_file]["instrument"]
                        if instr == query_instr:
                            folder = file.split('/')[-1][:-14] + '_Full'
                            filename = raw_files[raw_file]['filename']
                            wav_path = os.path.join(audiodata_path, "mixing-secrets", "wav", folder, filename)
                            copyfile(wav_path, f"out_{numb_found_stems}_.wav")
                            numb_found_stems += 1

            # for stem in stems:
            #     instr = stems[stem][key_name]
            #     if instr == query_instr:
            #         if dataset == 'slakh':
            #             if stems[stem]["audio_rendered"]:
            #                 track = file.split('/')[-1][:-5]
            #                 wav_path = os.path.join(audiodata_path, "slakh", "wav", track, "stems", stem + ".wav")
            #             else:
            #                 continue
            #         elif dataset == 'medleydb':
            #             track = file.split('/')[-1][:-14]
            #             filename = track + "_STEM_" + stem[1:] + ".wav"
            #             wav_path = os.path.join(audiodata_path, "medleydb", "wav", track, "STEMS", filename)
            #         elif dataset == 'mixing-secrets':
            #             folder = file.split('/')[-1][:-14] + '_Full'
            #             filename = stems[stem]['filename']
            #             wav_path = os.path.join(audiodata_path, "mixing-secrets", "wav", folder, filename)
            #         copyfile(wav_path, f"out_{numb_found_stems}_.wav")
            #         numb_found_stems += 1
            #         break

    if numb_found_stems > 0:
        print(f"found {numb_found_stems} stems with the query instrument.")
    else:
        print("didn't find a stem containing the query instrument!")


if __name__ == "__main__":
    main()