"""
script to play around with roc auc score, labelbinarizer and other stuff
"""
from cProfile import label
import os
import yaml
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import soundfile
from glob import glob
import librosa
import pyloudnorm as pyln
import matplotlib.pyplot as plt


def try_roc_auc():
    # example with 3 samples and 4 classes
    gt_array = np.array([[1,1,0,1],[0,0,1,0],[0,0,0,0]])
    #gt_array = np.array([[0,1,0,1],[0,0,1,0],[0,0,1,1]]) #value error: only one class in ytrue ...
    # for every class there must be at least one 0 and one 1
    est_array = np.array([[0.9,0.1,0.4,0.1],[0,0.6,0.3,0.1],[0.6,0.2,0.7,0.4]])
    roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
    print(roc_aucs)


def try_labelbinarizer():
    all_tags = ['e', 'b', 'c', 'd']
    tags = ['b', 'd', 'b']
    lb = LabelBinarizer().fit(all_tags)
    #mlb = MultiLabelBinarizer().fit(all_tags)
    lb_binary_tags = lb.transform(list(set(tags)))  #casting to set -> removes duplicates
    #mlb_binary_tags = mlb.transform(tags)
    #if np.array_equal(lb_binary_tags, mlb_binary_tags):
    #    print('arrays are equal!')
    
    print(lb_binary_tags)
    sum_tags = np.sum(lb_binary_tags, axis=0)
    pass


def load_activation_conf():
    import glob
    #files = glob.glob('/clusterFS/home/student/hbradl/instrument-recognition/data/metadata/raw/medleydb_activation_conf/*.lab')
    path = '/clusterFS/home/student/hbradl/instrument-recognition/data/metadata/raw/medleydb_activation_conf/MichaelKropf_AllGoodThings_ACTIVATION_CONF.lab'
    hop_size = 2048 / 44100 #s
    input_length = 5  #s
    numb_rows = round(input_length / hop_size)
    act_conf = pd.read_csv(path)

    rand_start_idx = 604000
    sr = 32000
    start_time = rand_start_idx / sr

    df_idx_start = round(start_time / hop_size)
    df_idx_end = round((start_time + input_length) / hop_size)

    stem = 1
    print(act_conf.iloc[df_idx_start:df_idx_end, stem])
    mean_activation = act_conf.iloc[df_idx_start:df_idx_end, stem].mean()
    print(mean_activation)


def check_if_activation_conf_file_exists():
    import glob
    act_conf_path = '/clusterFS/home/student/hbradl/instrument-recognition/data/metadata/raw/medleydb_activation_conf'
    metadata_path = '/clusterFS/home/student/hbradl/instrument-recognition/data/metadata/raw/medleydb'
    act_conf_files = glob.glob(os.path.join(act_conf_path, '*.lab'))
    metadata_files = glob.glob(os.path.join(metadata_path, '*.yaml'))

    count = 0
    for file in metadata_files:
        song_name = file.split('/')[-1][:-14]
        lab_fn = song_name + '_ACTIVATION_CONF.lab'
        lab_path = os.path.join(act_conf_path, lab_fn)
        if lab_path not in act_conf_files:
            count += 1
            print(song_name)
        
    print(count)

def replace_dashes_with_underscore():
    instr_classes = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']
    new_classes = []
    for string in instr_classes:
        new_classes.append(string.replace('---', '_'))
    print(new_classes)


def try_augmentations():
    from audiomentations import (Compose, ApplyImpulseResponse, TimeStretch, PitchShift, 
                            Gain, PeakingFilter, LowShelfFilter, HighShelfFilter)
    samplerate = 32000
    p_augment = 1.0  # probability that augmentation is applied
    filepath = '/clusterFS/home/student/hbradl/data/preprocessed/medleydb/wav/Lushlife_ToynbeeSuite/Lushlife_ToynbeeSuite_MIX.wav'
    audio, sr = soundfile.read(filepath, start=samplerate*60*3, frames=samplerate*5, dtype='float32')
    assert sr == samplerate
    ir_path = '/clusterFS/home/student/hbradl/instrument-recognition/irs'
    augment = Compose([ApplyImpulseResponse(ir_path, p=0, leave_length_unchanged=True),
                        TimeStretch(min_rate=0.8, max_rate=1.5, p=0.8),
                        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
                        Gain(min_gain_in_db=-8, max_gain_in_db=0, p=0.8),
                        PeakingFilter(min_center_freq=50, max_center_freq=8000, min_gain_db=-12,
                        max_gain_db=12, min_q=0.5, max_q=5, p=0.8),
                        HighShelfFilter(min_center_freq=300, max_center_freq=7500, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=0.8),
                        LowShelfFilter(min_center_freq=50, max_center_freq=4000, min_gain_db=-12, 
                        max_gain_db=12, min_q=0.1, max_q=0.999, p=0.8)],
                        p=p_augment, shuffle=True)

    audio = augment(audio, sample_rate=samplerate)

    soundfile.write("augmented_audio.wav", audio, samplerate)


def ir_preprocessing():
    ir_dir = '/clusterFS/home/student/hbradl/instrument-recognition/irs'
    new_dir = '/clusterFS/home/student/hbradl/instrument-recognition/irs_new/'
    if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    files = glob(os.path.join(ir_dir, '*.wav'))
    for file in files:
        x, sr = librosa.core.load(file, sr=32000, mono=True, res_type='kaiser_best')
        assert sr == 32000
        new_path = new_dir + file.split('/')[-1]
        soundfile.write(new_path, x, sr, 'PCM_16')


def load_yaml():
    with open("./../training/taxonomy.yaml", 'r') as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    print(dict)


def mapping():
    with open("./../training/taxonomy.yaml", 'r') as f:
        taxonomy_dict = yaml.load(f, Loader=yaml.FullLoader)

    #labels = ['Handbells', 'DrumSamples_HiHat', 'DrumsSideR', 'DrumSamples_Toms', 'Trumpet', 'Percussion', 'KickInside', 'violin', 'LeadVoxDouble', 'Anvil', 'FloorTom', 'Kalimba', 'SFX', 'SynthFuzz', 'GongRoll', 'DrumRoom', 'CymbalSample', 'DrumsRoomLoFi', 'Brushes', 'SynthFX', 'Timbale', 'FrenchHorn', 'female speaker', 'SynthFill', 'Bass', 'female singer', 'Clarinet', 'marimba', 'Drumkit_SnareMic', 'Sitar', 'SynthS', 'Organ', 'Harp', 'SnareRimOD_Overheads', 'SchoolBell', 'Marimba', 'SynthPad', 'Break', 'DjembeFX', 'DrumAmb', 'SynthGtr', 'OverheadsOH Stereo MK', 'SynthGtrDI', 'electric bass', 'DrumSamples_Snare', 'LeadVoxDT', 'Tamb', 'HatOpen', 'Wineglasses', 'Cellos', 'StringsPizz', 'Handclap', 'conga', 'male speaker', 'SampleBrass', 'distorted electric guitar', 'SnareRimOD_Room', 'KickBack', 'Drumkit_Tom', 'Drumkit_KickIn', 'DrumsRoomStereo', 'Violin', 'female singer sample', 'Cowbell', 'Wurlitzer', 'Viola', 'SynthRiff', 'Bongos', 'GlitchSFX', 'LowTom', 'ElecGtrLoop', 'Toms', 'HiHatOpen', 'Sticks', 'Crash', 'BassSub', 'KickFill', 'Dobro', 'HandBells', 'Gtr', 'HarpHarmonics', 'Paliteo', 'Congas', 'Breaks', 'SynthReverse', 'Kick', 'Saxophone', 'Rhodes', 'Strings_Ensemble', 'tambourine', 'DrumsRoomMono', 'Drumkit_Hihat', 'Tome', 'Gangs', 'Glockenspiel', 'Spot_LowStrings', 'DrumSamples_Cymbals', 'Claps', 'Overheads', 'SynthChoir', 'acoustic guitar', 'Ukelele', 'Ride', 'Loop', 'ElectroChoir', 'ShakerLoop', 'HiHatSample', 'male singer', 'Maracas', 'MalaysianDjembe', 'DrumsSideL', 'Hi-Hat', 'Drums', 'Bass samples', 'SFXHit', 'CymbalsOD_Room', 'saxophone', 'KickOut', 'SubDrop', 'Drumkit_RoomStereo', 'SynthsAndSFX', 'Shakers', 'Drumkit_KickSub', 'TomRim', 'PianoDry', 'Violins', 'Pair_OverheadORTF', 'HiHatOD', 'CymbalRolls', 'Piano', 'Psaltery', 'FingerCymbals', 'SynthSFX', 'electric piano', 'SnareTop', 'Bombo', 'TubularBells', 'SampleChoir', 'CymbalRoll', 'KickIn', 'ChinaCymbal', 'Triangle', 'Drumkit_RoomMono', 'Clavinet', 'HatClosed', 'AmbStereo', 'Clav', 'BassDI', 'Cello', 'Trombone', 'male screamer', 'SnareMic', 'male rapper', 'TomSample', 'Click', 'Ebow', 'SynthMel', 'RackTom', 'ReverseCymbal', 'distorted electric guitarRevFX', 'Scream', 'PadSFX', 'HatSample', 'Kicks', 'KickSub', 'SnareRimOD', 'SynthArp', 'KickTrigger', 'drum set', 'AcousticGtr', 'Strings', 'PianoRevSFX', 'SnareDown', 'Keys', 'Strings_Soli', 'Accordion', 'Cunono', 'Flute', 'ElecNoise', 'Rainstick', 'Accordian', 'Mellotron', 'Pad', 'Clap', 'Strings_Staccato', 'BackingVox', 'bass synth', 'Tuba', 'SynthStabs', 'Fill', 'Sidestick', 'Cymbal', 'Daiko', 'LeadVox', 'HandDrum', 'Violin samples', 'AmbMono', 'SynthLead', 'BackingVoxSFX', 'clean electric guitar samples', 'HiHatClosed', 'Bass synth', 'Hammond', 'Bongo', 'Hit', 'Djembe', 'SynthBass', 'Spot_HighStrings', 'SFXTransition', 'DrumsAmbience', 'Drumkit_KickOut', 'Rim', 'TomLo', 'DrumMic', 'Mandolin', 'HiHat', 'RimShot', 'GtrRevSFXMic', 'SynthStrings', 'ElectroHit', 'SampleSFX', 'RomanWarDrum', 'KickSample', 'DrumsReverb', 'Overhead', 'DrumLoop', 'double bass', 'clean electric guitar', 'SnareFX', 'Snare', 'viola', 'StringPad', 'KickFront', 'HiHatPedal', 'Harmonica', 'CymbalsOD_Overheads', 'RevCrash', 'Cymbals', 'RevCymbal', 'Hat', 'Tub', 'Synth', 'SubBass', 'Shaker', 'BassPod', 'Drumkit_Overheads', 'BassSFX', 'Hihat', 'Guasa', 'SynthDT', 'feedback', 'harp Sample', 'Horns', 'TomHi', 'violins', 'Violas', 'ElectroPercussion', 'KickBeater', 'Tom', 'Fiddle', 'VinylNoise', 'cello', 'BrushKit', 'Sprinkles', 'Sample', 'Noise', 'clean electric guitar reverse', 'fx/processed sound', 'Viola samples', 'SnareSample', 'HiHatOD_Overheads', 'Tambourine', 'RoomLoFi', 'AmoMono', 'BassSynth', 'SnareBottom', 'piano', 'Tubas', 'Boom', 'DrumsOverhead', 'Ambience', 'acoustic guitar reverse', 'Bells', 'Woodblock', 'Snares', 'TrumpetSFX', 'SFXVox', 'ShareDown', 'DrumsRoom', 'Taiko', 'Main System', 'distorted electric guitar samples', 'Transition', 'SynthLoop', 'SnareUp', 'Cello samples', 'PianoSFX', 'SFXPad']
    labels = ['Tango Accordion', 'Piccolo', 'FX 3 (crystal)', 'Orchestra Hit', 'Timpani', 'FX 4 (atmosphere)', 'Lead 5 (charang)', 'Orchestral Harp', 'Drawbar Organ', 'Synth Strings 1', 'Pad 6 (metallic)', 'Synth Strings 2', 'Helicopter', 'Tremolo Strings', 'Lead 7 (fifths)', 'Tinkle Bell', 'Pad 8 (sweep)', 'FX 7 (echoes)', 'Honky-tonk Piano', 'Marimba', 'Bag pipe', 'Baritone Sax', 'Reed Organ', 'Woodblock', 'Synth Bass 2', 'Clavinet', 'Pad 2 (warm)', 'Pad 4 (choir)', 'Electric Bass (pick)', 'FX 8 (sci-fi)', 'Synth Drum', 'Viola', 'Shamisen', 'Celesta', 'French Horn', 'Shakuhachi', 'Muted Trumpet', 'Steel Drums', 'Electric Piano 1', 'Oboe', 'Clarinet', 'Pizzicato Strings', 'Church Organ', 'Guitar Fret Noise', 'Fretless Bass', 'Rock Organ', 'FX 1 (rain)', 'Pad 5 (bowed)', 'Music Box', 'Synth Brass 2', 'Vibraphone', 'Tenor Sax', 'Sitar', 'Lead 6 (voice)', 'Synth Brass 1', 'Distortion Guitar', 'Cello', 'Electric Bass (finger)', 'Voice Oohs', 'Lead 3 (calliope)', 'Guitar harmonics', 'Taiko Drum', 'Kalimba', 'Brass Section', 'Acoustic Guitar (steel)', 'Harpsichord', 'Electric Grand Piano', 'Fiddle', 'Drums', 'Ocarina', 'Glockenspiel', 'Slap Bass 2', 'Accordion', 'FX 2 (soundtrack)', 'Acoustic Grand Piano', 'Overdriven Guitar', 'Bright Acoustic Piano', 'String Ensemble 2', 'Percussive Organ', 'Trumpet', 'Pad 1 (new age)', 'Harmonica', 'Koto', 'Acoustic Bass', 'Whistle', 'Bird Tweet', 'Telephone Ring', 'Xylophone', 'Gunshot', 'Soprano Sax', 'String Ensemble 1', 'Lead 4 (chiff)', 'Dulcimer', 'Slap Bass 1', 'Electric Guitar (jazz)', 'Applause', 'Electric Piano 2', 'Trombone', 'Blown Bottle', 'Synth Bass 1', 'Violin', 'Seashore', 'Lead 8 (bass + lead)', 'English Horn', 'Banjo', 'Breath Noise', 'Recorder', 'Alto Sax', 'Shanai', 'Electric Guitar (muted)', 'Tuba', 'Synth Voice', 'FX 6 (goblins)', 'Contrabass', 'Agogo', 'Lead 2 (sawtooth)', 'Pad 7 (halo)', 'Acoustic Guitar (nylon)', 'Pad 3 (polysynth)', 'Electric Guitar (clean)', 'Tubular Bells', 'Flute', 'Melodic Tom', 'Bassoon', 'FX 5 (brightness)', 'Choir Aahs', 'Lead 1 (square)', 'Pan Flute', 'Reverse Cymbal']
    #labels = ['trumpet', 'alto saxophone', 'banjo', 'dizi', 'bongo', 'bassoon', 'chimes', 'accordion', 'male singer', 'toms', 'bamboo flute', 'clarinet', 'male speaker', 'yangqin', 'cello', 'whistle', 'guzheng', 'electric piano', 'electric bass', 'piano', 'cello section', 'castanet', 'sampler', 'horn section', 'baritone saxophone', 'synthesizer', 'harp', 'double bass', 'gong', 'liuqin', 'tabla', 'violin section', 'harmonica', 'brass section', 'scratches', 'viola section', 'trombone', 'erhu', 'gu', 'viola', 'electronic organ', 'zhongruan', 'claps', 'french horn', 'doumbek', 'piccolo', 'tambourine', 'cymbal', 'bass drum', 'guiro', 'flute section', 'sleigh bells', 'kick drum', 'vocalists', 'auxiliary percussion', 'tenor saxophone', 'Main System', 'french horn section', 'mandolin', 'trombone section', 'high hat', 'trumpet section', 'melodica', 'fx/processed sound', 'drum set', 'female singer', 'male rapper', 'snare drum', 'clarinet section', 'euphonium', 'oboe', 'shaker', 'glockenspiel', 'distorted electric guitar', 'violin', 'clean electric guitar', 'cowbell', 'flute', 'string section', 'vibraphone', 'cornet', 'drum machine', 'tuba', 'darbuka', 'cabasa', 'tack piano', 'acoustic guitar', 'timpani', 'oud', 'lap steel guitar', 'soprano saxophone', 'bass clarinet']
    for label in labels:
        found_label = False
        for key0, value0 in taxonomy_dict.items():
            if isinstance(value0, dict):
                for key1, value1 in value0.items():
                    if value1 is None:
                        continue
                    if isinstance(value1, dict):
                        for key2, value2 in value1.items():
                            if value2 is None:
                                continue
                            if label in value2:
                                found_label = True
                    else:
                        if label in value1:
                            found_label = True
            else:
                if label in value0:
                    found_label = True

        if not found_label:
            print(label)


def features_silence():
    samplerate = 32000
    filepath = '/srv/ALL/datasets/preprocessed/medleydb/wav/Lushlife_ToynbeeSuite/Lushlife_ToynbeeSuite_MIX.wav'
    audio, sr = soundfile.read(filepath, start=samplerate*60*3, frames=samplerate*5, dtype='float32')
    mfccs = librosa.feature.mfcc(audio, sr=samplerate, n_mfcc=12)
    mean_mfccs = np.mean(mfccs, axis=1)
    audio = audio * 0.00995
    loudn_meter = pyln.Meter(samplerate)
    loudness = loudn_meter.integrated_loudness(audio)
    mfccs2 = librosa.feature.mfcc(audio, sr=samplerate, n_mfcc=12)
    mean_mfccs2 = np.mean(mfccs2, axis=1)
    print(loudness)
    print(mean_mfccs)


def plot_table():
    fig, ax =plt.subplots(1,1)
    data=[[1,2,3],
        [5,6,7]]
    column_labels=["Column 1", "Column 2", "Column 3"]
    row_labels=["Target", "Estimated"]
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,rowLabels=row_labels, colLabels=column_labels,loc="center")

    plt.savefig("example_table.png")
    plt.show()


def plot_metrics():
    '''plot roc-auc, pr-auc and test loss
    for various values for p_mult_songs'''
    p_mult_songs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    #test seed 123
    roc_auc = [0.935075, 0.935575, 0.934925, 0.930775, 0.924975, 0.91745]
    pr_auc = [0.81795, 0.821625, 0.82365, 0.818775, 0.811125, 0.79595]
    test_loss = [0.36345, 0.358675, 0.357425, 0.36075, 0.368025, 0.3888]

    #test seed 124
    # roc_auc = [0.933875, 0.93325, 0.932975, 0.929125, 0.92215, 0.916975]
    # pr_auc = [0.815375, 0.8164, 0.81545, 0.81575, 0.805325, 0.792825]
    # test_loss = [0.3584, 0.358675, 0.35685, 0.3583, 0.3658, 0.385525]

    fig, ax =plt.subplots(3,1)
    ax[0].plot(p_mult_songs, roc_auc)
    ax[0].set(xlabel=r"$p_{musical}$", ylabel="ROC-AUC")
    ax[1].plot(p_mult_songs, pr_auc)
    ax[1].set(xlabel=r"$p_{musical}$", ylabel="PR-AUC")
    ax[2].plot(p_mult_songs, test_loss)
    ax[2].set(xlabel=r"$p_{musical}$", ylabel="Test Loss")
    plt.savefig("p_mult_songs-experiment.png")


def plot_threshold_dependent_metrics():
    '''plot precision, recall and f1-score for 
    various thresholds'''
    thresholds = np.round(np.arange(0.1, 0.75, 0.05), 2).tolist()

    #test seed 123456; thresholds 0.1 - 0.7
    f1_score = [0.6917, 0.7177, 0.7205, 0.7342, 0.7253, 0.7305, 0.7138, 0.6808, 0.6548, 0.6237, 0.5791, 0.5530, 0.5118]
    precision = [0.5705, 0.6284, 0.6623, 0.7056, 0.7469, 0.7882, 0.8262, 0.8394, 0.8572, 0.8495, 0.8582, 0.8753, 0.8514]
    recall = [0.9265, 0.8682, 0.8142, 0.7834, 0.7231, 0.6960, 0.6497, 0.5966, 0.5521, 0.5149, 0.4642, 0.4324, 0.3881]

    fig, ax = plt.subplots(3,1)
    ax[0].plot(thresholds, precision)
    ax[0].set(xlabel="Threshold", ylabel="Precision")
    ax[1].plot(thresholds, recall)
    ax[1].set(xlabel="Threshold", ylabel="Recall")
    ax[2].plot(thresholds, f1_score)
    ax[2].set(xlabel="Threshold", ylabel="F1-Score")
    plt.savefig("threshold-experiment.png")


def plot_mel_scale():
    freqs = np.arange(0, 14000, 10)
    mels = 2595 * np.log10(1 + (freqs)/700)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(freqs, mels)
    ax.grid()
    ax.set_xticks(np.arange(min(freqs), max(freqs)+1, 1000))
    ax.set_xlim(left=0, right=12000)
    ax.set_ylim(bottom=0, top=3500)
    ax.set(xlabel="Frequency in Hz", ylabel="Frequency in Mel")
    plt.savefig("mel-scale.png")


def try_zero_crossing_rate():
    sine = np.sin(np.linspace(0, 4 * 2 * np.pi, 20000))
    noise = np.random.rand(20000) - 0.5
    zero_cross_rate = librosa.feature.zero_crossing_rate(noise + 0.0001, frame_length=10, hop_length=10, center=False)
    zero_crossings = librosa.zero_crossings(noise)
    mean_zcr = np.mean(zero_cross_rate)
    mean_zcr_2 = np.count_nonzero(zero_crossings) / 20000
    zcr_sum = np.sum(zero_cross_rate)
    plt.plot(noise)
    plt.savefig("zero_cross_rate.png")


def plot_loss_curves():
    import json
    folder = './../../misc/tensorboard-graphs/classification/transfer-learning-exp'

    #train loss
    with open(os.path.join(folder, 'from-scratch-trainloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    from_scratch_trainloss = array[:, 2]

    with open(os.path.join(folder, 'frozen-backbone-trainloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    frozen_trainloss = array[:, 2]

    with open(os.path.join(folder, 'finetuned-trainloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    finetuned_trainloss = array[:, 2]

    #valid loss
    with open(os.path.join(folder, 'from-scratch-validloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    from_scratch_validloss = array[:, 2]

    with open(os.path.join(folder, 'frozen-backbone-validloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    frozen_validloss = array[:, 2]

    with open(os.path.join(folder, 'finetuned-validloss.json')) as json_file:
        data = json.load(json_file)
    array = np.array(data)
    finetuned_validloss = array[:, 2]

    train_epochs = np.arange(1, len(from_scratch_trainloss)+1)
    valid_epochs = np.arange(1, len(from_scratch_validloss)+1)


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    ax[0].plot(train_epochs, from_scratch_trainloss, label='from-scratch')
    ax[0].plot(train_epochs, frozen_trainloss, label='frozen-backbone')
    ax[0].plot(train_epochs, finetuned_trainloss, label='finetuned')
    ax[0].set(xlabel="Epoch", ylabel="Train Loss")
    ax[0].legend()
    ax[1].plot(valid_epochs, from_scratch_validloss, label='from-scratch')
    ax[1].plot(valid_epochs, frozen_validloss, label='frozen-backbone')
    ax[1].plot(valid_epochs, finetuned_validloss, label='finetuned')
    ax[1].set(xlabel="Epoch", ylabel="Valid Loss")
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("transfer-learning-exp.png")


if __name__ == "__main__":
    #try_labelbinarizer()
    #try_roc_auc()
    #load_activation_conf()
    #check_if_activation_conf_file_exists()
    #replace_dashes_with_underscore()
    #try_augmentations()
    #ir_preprocessing()
    #load_yaml()
    #mapping()
    #features_silence()
    #plot_table()
    #plot_metrics()
    #plot_threshold_dependent_metrics()
    #plot_mel_scale()
    #try_zero_crossing_rate()
    plot_loss_curves()
