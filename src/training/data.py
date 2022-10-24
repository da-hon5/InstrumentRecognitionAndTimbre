"""
this file contains all the datasets
"""
import os
import yaml
import csv
from glob import glob
from collections import defaultdict
import numpy as np
import torch
import librosa
import soundfile
from sklearn.preprocessing import LabelBinarizer
import pyloudnorm as pyln
from audiomentations import (Compose, TimeStretch, PitchShift, 
                            Gain, PeakingFilter, LowShelfFilter, HighShelfFilter)

import utils


class Augmentation:
    def __init__(self, aug_config):
        self.p_augment = aug_config.p_augment
        self.p_imp_res = aug_config.p_imp_res
        self.p_time_stretch = aug_config.p_time_stretch
        self.p_pitch_shift = aug_config.p_pitch_shift
        self.p_rand_gain = aug_config.p_rand_gain
        self.p_peaking_filter = aug_config.p_peaking_filter
        self.p_lowshelf_filter = aug_config.p_lowshelf_filter
        self.p_highshelf_filter = aug_config.p_highshelf_filter

        self.all = Compose([TimeStretch(min_rate=0.8, max_rate=1.5, p=self.p_time_stretch),
                        PitchShift(min_semitones=-4, max_semitones=4, p=self.p_pitch_shift),
                        Gain(min_gain_in_db=-12, max_gain_in_db=0, p=self.p_rand_gain),
                        PeakingFilter(min_center_freq=50, max_center_freq=8000, min_gain_db=-12,
                        max_gain_db=12, min_q=0.5, max_q=5, p=self.p_peaking_filter),
                        HighShelfFilter(min_center_freq=300, max_center_freq=7500, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=self.p_highshelf_filter),
                        LowShelfFilter(min_center_freq=50, max_center_freq=4000, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=self.p_lowshelf_filter)],
                        p=self.p_augment, shuffle=True)

        self.only_filter = Compose([Gain(min_gain_in_db=-12, max_gain_in_db=0, p=self.p_rand_gain),
                        PeakingFilter(min_center_freq=50, max_center_freq=8000, min_gain_db=-12,
                        max_gain_db=12, min_q=0.5, max_q=5, p=self.p_peaking_filter),
                        HighShelfFilter(min_center_freq=300, max_center_freq=7500, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=self.p_highshelf_filter),
                        LowShelfFilter(min_center_freq=50, max_center_freq=4000, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=self.p_lowshelf_filter)],
                        p=self.p_augment, shuffle=True)


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, medleydb_dataset, mixingsecrets_dataset, slakh_dataset, data_config, split, samples_per_epoch):
        self.medleydb = medleydb_dataset
        self.mixingsecrets = mixingsecrets_dataset
        self.slakh = slakh_dataset
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.p_medleydb = data_config.p_medleydb
        self.p_mixingsecrets = data_config.p_mixingsecrets
        self.p_slakh = data_config.p_slakh

        if self.p_medleydb + self.p_mixingsecrets + self.p_slakh != 1:
            raise ValueError("Probabilities have to sum up to 1!")

        if self.medleydb.num_files == 0:
            print("Warning: MedleyDB has 0 songs --> setting p_medleydb = 0.0!")
            self.p_medleydb = 0.0
            self.p_mixingsecrets = 0.5
            self.p_slakh = 0.5
        elif self.mixingsecrets.num_files == 0:
            print("Warning: Mixingsecrets has 0 songs --> setting p_mixingsecrets = 0.0!")
            self.p_mixingsecrets = 0.0
            self.p_medleydb = 0.5
            self.p_slakh = 0.5
	
    def __getitem__(self, index):
        if self.split in ['valid', 'test']:
            np.random.seed(index)
            
        dataset = np.random.choice(['medleydb', 'mixing-secrets', 'slakh'],
                                p=[self.p_medleydb, self.p_mixingsecrets, self.p_slakh])

        if dataset == 'medleydb':
            return self.medleydb[index]
        elif dataset == 'mixing-secrets':
            return self.mixingsecrets[index]
        elif dataset == 'slakh':
            return self.slakh[index]


    def __len__(self):
        return self.samples_per_epoch



class MultiTrackDataset(torch.utils.data.Dataset):
    """ parent class for all the multi-track datasets """
    def __init__(self, data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track):
        self.root = data_config.data_path
        self.dataset = data_config.dataset
        self.split = split
        self.timb_feat = timb_feat
        self.chunks_per_track = chunks_per_track
        self.samplerate = data_config.samplerate
        self.input_length = data_config.model_input_length * self.samplerate
        self.max_num_sources = data_config.max_num_sources
        self.p_single_source = data_config.p_single_source
        self.p_mult_songs = data_config.p_multiple_songs
        self.p_skip_percussion = data_config.p_skip_percussion
        self.p_skip_plucked_str = data_config.p_skip_plucked_str
        self.n_mfccs = data_config.n_mfccs
        self.predict_loudness = data_config.predict_loudness
        self.loudn_threshold = data_config.loudn_threshold
        self.loudn_meter = pyln.Meter(self.samplerate)
        self.instr_families = utils.load_classes(type='instr_families')
        self.instr_classes = utils.load_classes(type='instr_classes')
        self.mlb = LabelBinarizer().fit(self.instr_families + self.instr_classes)
        self.augmentation = Augmentation(aug_config)
        self.file_dict = {}
        self.num_files = 0
        self.samples_per_epoch = samples_per_epoch
        self.index_count = defaultdict(int)
        self.target_class = data_config.target_class


    def get_wav_path(self, song_name, source='mix'):
        """ returns path of a wav-file (has to be implemented in child classes) """
        raise NotImplementedError


    def get_metadata_files(self):
        """ returns metadata files for training (has to be implemented in child classes) """
        raise NotImplementedError


    def get_song_info(self, doc):
        """ gets song info from metadata file (has to be implemented in child classes) """
        raise NotImplementedError


    def get_docs(self, metadata_files):
        docs = []
        for file in metadata_files:
            with open(file, 'r') as f:
                docs.append(yaml.load(f, Loader=yaml.FullLoader))
        return docs


    def get_file_dict(self):
        file_dict = {}
        metadata_files = self.get_metadata_files()
        metadata_files.sort()
        docs = self.get_docs(metadata_files)
        count = 0
        for doc in docs:
            song_name, source_filenames, instr_fams, instr_classes, instr_subclasses = self.get_song_info(doc)
            if self.target_class in (instr_fams+instr_classes+instr_subclasses) or self.target_class == 'all-classes':
                file_dict[count] = {
                    'song_name': song_name, 
                    'source_filenames': source_filenames, 
                    'instr_families': instr_fams,
                    'instr_classes': instr_classes,
                    'instr_subclasses': instr_subclasses}

                count += 1
            
        return file_dict


    def get_rand_num_sources(self, index, use_mult_songs):
        if use_mult_songs:
            max_num_sources = self.max_num_sources
        else:
            max_num_sources = len(self.file_dict[index]['source_filenames'])

        return np.random.randint(low=1, high=max_num_sources+1)


    def compute_timbre_features(self, audio):
        """ returns vector with multiple concatenated timbre descriptors """
        #TODO: discard 0th mfcc coefficient (total energy)?
        feature_vector = []

        # loudness
        if self.predict_loudness:
            loudness = self.loudn_meter.integrated_loudness(audio)
            feature_vector = np.append(feature_vector, loudness)
            if loudness == float("-inf") or loudness < -100:
                raise RuntimeError("Loudness smaller than -100 LUFS detected!")

        # only compute timbre descriptors of non-silent slices
        n_slices = 8
        threshold = 0.0003  # -70dB = ~0.0003; -80dB = 0.0001
        rms_array = []
        slices = np.split(audio, n_slices)
        for slice in slices:
            rms = np.sqrt(np.mean(np.square(slice)))
            rms_array.append(rms)
        
        max_index = np.argmax(rms_array)
        non_silent_slices = [t[1] for t in zip(rms_array, slices) if t[0] > threshold]
        if len(non_silent_slices) == 0:
            non_silent_slices = slices[max_index]
        non_silent_audio = np.concatenate(non_silent_slices)

        # mfccs
        mfccs = librosa.feature.mfcc(non_silent_audio, sr=self.samplerate, n_mfcc=self.n_mfccs+1)
        feature_vector = np.append(feature_vector, np.median(mfccs, axis=1)[1:])
        feature_vector = np.append(feature_vector, utils.iqr(mfccs, axis=1)[1:])
        
        # other features
        spec_cent = librosa.feature.spectral_centroid(non_silent_audio, sr=self.samplerate)
        zero_cross_rate = librosa.feature.zero_crossing_rate(non_silent_audio + 0.0001)
        spec_flatness = librosa.feature.spectral_flatness(non_silent_audio)
        feature_vector = np.append(feature_vector, np.median(spec_cent, axis=1))
        feature_vector = np.append(feature_vector, utils.iqr(spec_cent, axis=1))
        feature_vector = np.append(feature_vector, np.median(zero_cross_rate, axis=1))
        feature_vector = np.append(feature_vector, utils.iqr(zero_cross_rate, axis=1))
        feature_vector = np.append(feature_vector, np.median(spec_flatness, axis=1))
        feature_vector = np.append(feature_vector, utils.iqr(spec_flatness, axis=1))
        
        return feature_vector


    def is_silent(self, audio):
        """ returns True if audio is silent (below threshold) """
        #rms = np.sqrt(np.mean(np.square(audio)))
        loudn = self.loudn_meter.integrated_loudness(audio)
        if loudn > self.loudn_threshold:
            return False
        return True


    def get_rand_source(self, index):
        #TODO: remove used sources/instr_classes from available sources ???
        song_name = self.file_dict[index]['song_name']
        all_sources = self.file_dict[index]['source_filenames']
        source_idx = np.random.randint(len(all_sources))
        source = all_sources[source_idx]
        instr_fam = self.file_dict[index]['instr_families'][source_idx]
        instr_class = self.file_dict[index]['instr_classes'][source_idx]
        instr_subclass = self.file_dict[index]['instr_subclasses'][source_idx]

        return song_name, source, instr_fam, instr_class, instr_subclass

    
    def get_rand_start_index(self, wav_path):
        song_length = utils.get_songlength(wav_path)
        start = int(np.floor(np.random.random(1) * (song_length - self.input_length)))
        return start


    def get_mix_dict(self, index=None, load_whole_song=False,
                    use_mult_songs=None, use_single_source=None, skip_percussion=None,
                    skip_plucked_str=None):
        """ creates new audio mix """
        num_sources = 1 if use_single_source else self.get_rand_num_sources(index, use_mult_songs)

        mix_dict = {} # dict of dicts
        used_sources = defaultdict(list)
        source_count = 0
        loop_count = 0
        while source_count < num_sources:
            if loop_count == 100:
                if source_count > 0:
                    break
                else:
                    return None
            
            loop_count += 1

             # choose new rand index when using multiple songs
            if use_mult_songs:
                index = np.random.randint(self.num_files)

            # choose a source
            song_name, source, instr_fam, instr_class, instr_subclass = self.get_rand_source(index)

            # choose new source when certain conditions are true
            # if instr_class in used_instr_classes:
            #     continue
            if song_name in used_sources.keys():
                if source in used_sources[song_name]:
                    continue
            if instr_fam is None:
                continue
            if (instr_fam == 'percussion') and skip_percussion:
                continue
            if (instr_fam == 'plucked-str') and skip_plucked_str:
                continue

            # load and augment wav-file
            wav_path = self.get_wav_path(song_name, source)
            if load_whole_song:
                audio, sr = soundfile.read(wav_path, dtype='float32')
            else:
                if use_mult_songs or source_count == 0:
                    start = self.get_rand_start_index(wav_path)
                audio, sr = soundfile.read(wav_path, start=start, frames=self.input_length, dtype='float32')

            if use_mult_songs or (num_sources == 1):
                audio = self.augmentation.all(audio, sample_rate=self.samplerate)
            else:
                audio = self.augmentation.only_filter(audio, sample_rate=self.samplerate)

            if not self.is_silent(audio):
                mix_dict[source_count] = {'audio': audio, 
                                        'instr_fam': instr_fam, 
                                        'instr_class': instr_class,
                                        'instr_subclass': instr_subclass}
                source_count += 1
                used_sources[song_name].append(source)

        return mix_dict


    def get_audio_mix(self, mix_dict, start_frame=None, stop_frame=None):
        """returns sum of all audio signals in mix_dict to feed the neural network"""
        audio_mix = np.zeros(1, dtype='float32')
        target_is_silent = False
        for value in mix_dict.values():
            if isinstance(start_frame, int) and isinstance(start_frame, int):
                audio = value['audio'][start_frame:stop_frame]
            else:
                audio = value['audio']
                
            if (self.target_class == value['instr_fam']
                or self.target_class == value['instr_class']
                or self.target_class == value['instr_subclass']):
                if self.is_silent(audio):
                    target_is_silent = True

            audio_mix = np.add(audio_mix, audio)

        audio_mix, is_clipping = utils.peak_normalize_if_clipping(audio_mix)
        # if is_clipping:
        #     print('is clipping')

        return audio_mix, target_is_silent

    
    def get_mix_tags(self, mix_dict):
        """returns targets for classification"""
        mix_tags = []
        for value in mix_dict.values():
            if value['instr_fam'] in self.instr_families:
                mix_tags.append(value['instr_fam'])
            if value['instr_class'] in self.instr_classes:
                mix_tags.append(value['instr_class'])
            #TODO: also output subclass ???
            
        tags_binary = np.sum(self.mlb.transform(list(set(mix_tags))), axis=0)
        return tags_binary


    def get_timbre_feature_dict(self, mix_dict, start_frame=None, stop_frame=None):
        """returns targets for feature prediction"""
        #TODO: which vector to output if an instrument is not in the mix? mfccs all zero? loudness=-inf or -100(or another integer)???
        #TODO: actually it is enough to compute timbre features of the target class!
        if self.predict_loudness:
            timbre_feature_dict = {k: np.zeros(1 + 2*(self.n_mfccs + 3)) for k in (self.instr_families + self.instr_classes)}
            for value in timbre_feature_dict.values():
                value[0] = -70  # -70 LKFS = absolute loudness threshold
        else:
            timbre_feature_dict = {k: np.zeros(2*(self.n_mfccs + 3)) for k in (self.instr_families + self.instr_classes)}

        for fam in self.instr_families:
            audio_fam_mix = np.zeros(1, dtype='float32')
            for value in mix_dict.values():
                if value['instr_fam'] == fam:
                    if isinstance(start_frame, int) and isinstance(start_frame, int):
                        audio = value['audio'][start_frame:stop_frame]
                    else:
                        audio = value['audio']
                    audio_fam_mix = np.add(audio_fam_mix, audio)
            if (len(audio_fam_mix) == self.input_length) and (not self.is_silent(audio_fam_mix)):
                audio_fam_mix, _ = utils.peak_normalize_if_clipping(audio_fam_mix)  #is normalizing allowed here? --> wrong loudness computed?
                timbre_feature_dict[fam] = self.compute_timbre_features(audio_fam_mix)

        for clss in self.instr_classes:
            audio_class_mix = np.zeros(1, dtype='float32')
            for value in mix_dict.values():
                if value['instr_class'] == clss:
                    if isinstance(start_frame, int) and isinstance(start_frame, int):
                        audio = value['audio'][start_frame:stop_frame]
                    else:
                        audio = value['audio']
                    audio_class_mix = np.add(audio_class_mix, audio)
            if (len(audio_class_mix) == self.input_length) and (not self.is_silent(audio_class_mix)):
                audio_class_mix, _ = utils.peak_normalize_if_clipping(audio_class_mix)  #is normalizing allowed here? --> wrong loudness computed?
                timbre_feature_dict[clss] = self.compute_timbre_features(audio_class_mix)

        return timbre_feature_dict


    def get_evaluation_batch(self, mix_dict):
        """ returns numpy array of multiple chunks 
        of the same audio file (used for validation and testing) """
        song_length = len(mix_dict[0]['audio'])
        hop = (song_length - self.input_length) // self.chunks_per_track
        audio_batch = np.ndarray(0)
        timbre_feature_dict_batch = {k: np.ndarray(0) for k in (self.instr_families+self.instr_classes)}
        batchsize = 0
        for i in range(self.chunks_per_track):
            audio_mix, target_is_silent = self.get_audio_mix(mix_dict, start_frame=i*hop, stop_frame=i*hop+self.input_length)
            # check if target_class is in the chunk (audio_mix) --> if not: discard chunk
            if not target_is_silent and not self.is_silent(audio_mix):
                #soundfile.write(f"./wav-files/{i}.wav", audio_mix, samplerate=self.samplerate)  #NOTE:  FOR DEBUGGING !!!!

                batchsize += 1
                audio_mix, _ = utils.peak_normalize_if_clipping(audio_mix)
                audio_batch = np.vstack((audio_batch, audio_mix)) if batchsize > 1 else audio_mix

                timbre_feature_dict = self.get_timbre_feature_dict(mix_dict, start_frame=i*hop, stop_frame=i*hop+self.input_length)
                for key in timbre_feature_dict_batch.keys():
                    timbre_feature_dict_batch[key] = np.vstack((timbre_feature_dict_batch[key], timbre_feature_dict[key])) if batchsize > 1 else timbre_feature_dict[key]
            #else:
                #soundfile.write(f"./wav-files/{i}_silent.wav", audio_mix, samplerate=self.samplerate)  #NOTE:  FOR DEBUGGING !!!!
            
        audio_batch = torch.from_numpy(audio_batch)
        
        tags_binary = self.get_mix_tags(mix_dict)
        tags_binary_batch = torch.tensor([tags_binary for _ in range(batchsize)])

        return audio_batch, tags_binary_batch, timbre_feature_dict_batch
    

    def is_target_class_in_mix_dict(self, mix_dict):
        for value in mix_dict.values():
            if (self.target_class == value['instr_fam']
                or self.target_class == value['instr_class']
                or self.target_class == value['instr_subclass']
                or self.target_class == 'all-classes'):
                return True
        return False


    def __getitem__(self, index):
        if self.split == "train":
            rnd_index = np.random.randint(self.num_files)
            use_mult_songs = np.random.choice([False, True], p=[1 - self.p_mult_songs, self.p_mult_songs])
            use_single_source = np.random.choice([False, True], p=[1 - self.p_single_source, self.p_single_source])
            skip_percussion = np.random.choice([False, True], p=[1 - self.p_skip_percussion, self.p_skip_percussion])
            skip_plucked_str = np.random.choice([False, True], p=[1 - self.p_skip_plucked_str, self.p_skip_plucked_str])

            mix_dict = self.get_mix_dict(index=rnd_index,
                                        load_whole_song=False,
                                        use_mult_songs=use_mult_songs,
                                        use_single_source=use_single_source,
                                        skip_percussion=skip_percussion,
                                        skip_plucked_str=skip_plucked_str)
            
            if mix_dict is None:
                return self.__getitem__(index) #NOTE: previous version: return self.__getitem__((index + 1) % self.__len__())

            if not self.is_target_class_in_mix_dict(mix_dict):
                return self.__getitem__(index)  #NOTE: previous version: return self.__getitem__((index + 1) % self.__len__())

            audio, target_is_silent = self.get_audio_mix(mix_dict)

            #soundfile.write("./wav-files/audio_mix.wav", audio, samplerate=self.samplerate)  #NOTE:  FOR DEBUGGING

            tag_binary = self.get_mix_tags(mix_dict)

            timbre_feature_dict = {}
            if self.timb_feat:
                timbre_feature_dict = self.get_timbre_feature_dict(mix_dict)

            return audio, tag_binary, timbre_feature_dict

        elif self.split in ["valid", "test"]:
            # prevent that an index is used too often during evaluation
            np.random.seed(index)
            avail_indices = np.array([k for k, v in self.index_count.items() if v < 15])
            if len(avail_indices) == 0:
                print("Warning: All songs from one dataset have been used 15 times during evaluation!")
            rnd_index = np.random.choice(avail_indices)
            self.index_count[rnd_index] += 1

            use_single_source = np.random.choice([False, True], p=[1 - self.p_single_source, self.p_single_source])
            mix_dict = self.get_mix_dict(index=rnd_index,
                                        load_whole_song=True,
                                        use_mult_songs=False, 
                                        use_single_source=use_single_source,
                                        skip_percussion=False,
                                        skip_plucked_str=False)

            if mix_dict is None:
                return self.__getitem__(index + 1)  #NOTE: previous version: return self.__getitem__((index + 1) % self.__len__())

            if not self.is_target_class_in_mix_dict(mix_dict):
                return self.__getitem__(index + 1)  #NOTE: previous version: return self.__getitem__((index + 1) % self.__len__())

            audio_batch, tags_binary_batch, timbre_feature_dict_batch = self.get_evaluation_batch(mix_dict)

            if audio_batch.size(dim=0) == 0:
                return self.__getitem__(index + 1)  #NOTE: previous version: return self.__getitem__((index + 1) % self.__len__())

            return audio_batch, tags_binary_batch, timbre_feature_dict_batch


    def __len__(self):
        return self.samples_per_epoch


class MedleydbDataset(MultiTrackDataset):
    def __init__(self, data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track=0):
        super().__init__(data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track)
        self.file_dict = self.get_file_dict()
        self.num_files = len(list(self.file_dict.keys()))
        self.index_count = {k:0 for k in range(self.num_files)}
        print(f"MedleyDB {split}: {self.num_files} Songs")
        

    def get_metadata_files(self):
        return glob(os.path.join("./../../data/splits/medleydb", self.split, "*.yaml"))

    
    def get_song_info(self, doc):
        source_filenames = []
        instr_fams = []
        instr_classes = []
        instr_subclasses = []
        song_name = doc["mix_filename"][0:-8]
        stems = doc["stems"]
        for stem in stems:
            raw_files = stems[stem]["raw"]

            for raw_file in raw_files:
                source_filenames.append(raw_files[raw_file]["filename"])
                instr_fams.append(raw_files[raw_file]["++FAMILY++"])
                instr_classes.append(raw_files[raw_file]["++CLASS++"])
                instr_subclasses.append(raw_files[raw_file]["++SUBCLASS++"])

        return song_name, source_filenames, instr_fams, instr_classes, instr_subclasses


    def get_wav_path(self, song_name, source='mix'):
        if source == 'mix':
            wav_path = os.path.join(self.root, "medleydb", "wav", song_name, song_name + "_MIX.wav")
        else:
            wav_path = os.path.join(self.root, 'medleydb', "wav", song_name, "RAW", source)
        return wav_path



class MixingSecretsDataset(MultiTrackDataset):
    def __init__(self, data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track=0):
        super().__init__(data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track)
        self.file_dict = self.get_file_dict()
        self.num_files = len(list(self.file_dict.keys()))
        self.index_count = {k:0 for k in range(self.num_files)}
        print(f"Mixing-Secrets {split}: {self.num_files} Songs")


    def get_metadata_files(self):
        return glob(os.path.join("./../../data/splits/mixing-secrets", self.split, "*.yaml"))


    def get_song_info(self, doc):
        source_filenames = []
        instr_fams = []
        instr_classes = []
        instr_subclasses = []
        song_name = doc["mix_filename"][0:-8]
        stems = doc["stems"]
        for stem in stems:
            raw_files = stems[stem]["raw"]

            for raw_file in raw_files:
                source_filenames.append(raw_files[raw_file]["filename"])
                instr_fams.append(raw_files[raw_file]["++FAMILY++"])
                instr_classes.append(raw_files[raw_file]["++CLASS++"])
                instr_subclasses.append(raw_files[raw_file]["++SUBCLASS++"])

        return song_name, source_filenames, instr_fams, instr_classes, instr_subclasses


    def get_wav_path(self, song_name, source='mix'):
        if source == 'mix':
            raise ValueError("No mixes available for mixing-secrets dataset!")
        else:
            wav_path = os.path.join(self.root, 'mixing-secrets', "wav", song_name + '_Full', source)
        return wav_path


class SlakhDataset(MultiTrackDataset):
    def __init__(self, data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track=0):
        super().__init__(data_config, aug_config, split, timb_feat, samples_per_epoch, chunks_per_track)
        self.file_dict = self.get_file_dict()
        self.num_files = len(list(self.file_dict.keys()))
        self.index_count = {k:0 for k in range(self.num_files)}
        print(f"Slakh {split}: {self.num_files} Songs")


    def get_metadata_files(self):
        return glob(os.path.join("./../../data/splits/slakh", self.split, "*.yaml"))
    

    def get_song_info(self, doc):
        song_name = doc["audio_dir"].split('/')[0]
        source_filenames = []
        instr_fams = []
        instr_classes = []
        instr_subclasses = []
        stems = doc["stems"]
        for stem in stems:
            if stems[stem]["audio_rendered"]:
                source_filenames.append(stem + ".wav")
                instr_fams.append(stems[stem]["++FAMILY++"])
                instr_classes.append(stems[stem]["++CLASS++"])
                instr_subclasses.append(stems[stem]["++SUBCLASS++"])

        return song_name, source_filenames, instr_fams, instr_classes, instr_subclasses


    def get_wav_path(self, song_name, source='mix'):
        if source == 'mix':
            wav_path = os.path.join(self.root, "slakh", "wav", song_name, "mix.wav")
        else:
            wav_path = os.path.join(self.root, 'slakh', "wav", song_name, "stems", source)
        return wav_path
    


class JamendoDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split, chunks_per_track=0):
        self.root = data_config.data_path
        self.split = split
        self.samplerate = data_config.samplerate
        self.input_length = data_config.model_input_length * self.samplerate
        self.classes = utils.load_classes(type='jamendo_classes')
        self.mlb = LabelBinarizer().fit(self.classes)
        self.chunks_per_track = chunks_per_track
        self.get_songlist()


    def get_songlist(self):
        META_PATH = './../../data/splits/jamendo/'
        if self.split == 'train':
            train_file = os.path.join(META_PATH, 'autotagging_top50tags-train.tsv')
            self.file_dict = self.get_file_dict(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'valid':
            valid_file = os.path.join(META_PATH,'autotagging_top50tags-validation.tsv')
            self.file_dict= self.get_file_dict(valid_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'test':
            test_file = os.path.join(META_PATH, 'autotagging_top50tags-test.tsv')
            self.file_dict= self.get_file_dict(test_file)
            self.fl = list(self.file_dict.keys())
        else:
            print('Split should be one of [train, valid, test]')


    def get_file_dict(self, tsv_file):
        tracks = {}
        with open(tsv_file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                track_id = row[0]
                tags = [tag.replace('---', '_') for tag in row[5:]]  # yaml config file doesn't like dashes
                tracks[track_id] = {
                    'path': row[3].replace('.mp3', '.wav'),
                    'tags': tags,
                    'dur_samples': int(float(row[4])) * self.samplerate
                }
        return tracks


    def __getitem__(self, index):
        if self.split == 'train':
            npy, tag_binary = self.get_train_sample(index)
        elif self.split in ['valid', 'test']:
            npy, tag_binary = self.get_evaluation_sample(index)
        return npy, tag_binary, 0


    def get_train_sample(self, index):
        """ returns a random chunk of audio (used for training) as numpy array """
        track = self.fl[index]
        filename = self.file_dict[track]['path']
        wav_path = os.path.join(self.root, 'jamendo', 'wav', filename[3:])
        song_length = utils.get_songlength(wav_path)
        random_idx = int(np.floor(np.random.random(1) * (song_length - self.input_length)))
        audio, sr = soundfile.read(wav_path, start=random_idx, frames=self.input_length, dtype='float32')
        assert sr == self.samplerate
        tags_binary = np.sum(self.mlb.transform(self.file_dict[track]['tags']), axis=0)
        return audio, tags_binary


    def get_evaluation_sample(self, index):
        """ returns numpy array of multiple chunks of audio from the same song 
        (used for validation and testing """
        track = self.fl[index]
        filename = self.file_dict[track]['path']
        wav_path = os.path.join(self.root, 'jamendo', 'wav', filename[3:])

        song_length = utils.get_songlength(wav_path)
        hop = (song_length - self.input_length) // self.chunks_per_track
        audio_array = torch.zeros(self.chunks_per_track, self.input_length)
        for i in range(self.chunks_per_track):
            audio, sr = soundfile.read(wav_path, start=i*hop, frames=self.input_length, dtype='float32')
            assert sr == self.samplerate
            audio_array[i] = torch.Tensor(audio).unsqueeze(0)

        tags_binary = np.sum(self.mlb.transform(self.file_dict[track]['tags']), axis=0)
        tags_binary = torch.tensor([tags_binary for _ in range(self.chunks_per_track)])
        return audio_array, tags_binary


    def __len__(self):
        return len(self.fl)
