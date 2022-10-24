import os
import yaml
from glob import glob
from sklearn.manifold import TSNE
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile
import librosa
import tqdm
import torchaudio
import pyloudnorm as pyln
import essentia.standard as ess
import soundfile
import plotly.express as px
from torch.utils.tensorboard import SummaryWriter

#TODO: make t-sne plot for every family !!!

audiodata_path = '/srv/ALL/datasets/preprocessed'
samplerate = 32000

def compute_timbre_features(audio, sr, n_mfccs):
    """ returns vector with muliptle concatenated timbre descriptors """
    framesize = 2048 #use bigger framesize (better freq resolution) !!!!
    hopsize = 512
    spec_size = int(framesize / 2 + 1)
    #TODO: exclude zeroth mfcc coefficient? --> represents average energy of signal

    window = ess.Windowing(type='hann', size=framesize) #blackman-harris window for SpectralPeaks()?
    spectrum = ess.Spectrum(size=framesize)
    spec_peaks = ess.SpectralPeaks(sampleRate=sr, magnitudeThreshold=-60, maxPeaks=100)
    yin = ess.PitchYinFFT(frameSize=framesize, sampleRate=sr)
    harm_peaks = ess.HarmonicPeaks(maxHarmonics=20, tolerance=0.2)
    inharm = ess.Inharmonicity()
    odd_to_even = ess.OddToEvenHarmonicEnergyRatio()
    tristim = ess.Tristimulus()
    mfcc = ess.MFCC(inputSize=spec_size, numberCoefficients=n_mfccs, sampleRate=sr)

    mfcc_array = []
    inharmonicities = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
        spec = spectrum(window(frame))
        spec_db = 20 * np.log10(spec + 10**-20) #add small number to avoid log(0)
        sp_freqs, sp_magn = spec_peaks(spec_db)
        f_0, _ = yin(spec)
        if len(sp_freqs) > 0:
            if sp_freqs[0] == 0:
                sp_freqs = sp_freqs[1:]
                sp_magn = sp_magn[1:]
        harm_freqs, harm_magn = harm_peaks(sp_freqs, sp_magn, f_0) 
        #If a particular harmonic was not found among spectral peaks, its ideal frequency value is output together with 0 magnitude
        inharmonicity = inharm(harm_freqs, harm_magn)
        inharmonicities.append(inharmonicity)
        #odd_to_even_harm_energy_ratio = odd_to_even(harm_freqs, harm_magn)
        #tristimulus = tristim(harm_freqs, harm_magn)
        _, mfccs = mfcc(spec)
        mfcc_array.append(mfccs)

    mfccs_mean = np.mean(mfcc_array, axis=0)[1:]

    librosa_mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfccs)
    librosa_mfccs_mean = np.mean(librosa_mfccs, axis=1)[1:]

    # #plot spectrogram for debugging
    # x_axis = np.linspace(0, sr/2, num=len(spec))
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(x_axis, spec)
    # ax[1].plot(x_axis, spec_db)
    # plt.savefig('spectrogram.png')
    # plt.close()

    # #plot mfccs for debugging
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(mfccs_mean)
    # ax[1].plot(librosa_mfccs_mean)
    # plt.savefig('mfccs-librosa_vs_essentia.png')
    # plt.close()

    # #save audio chunk for debugging
    # soundfile.write('audio-chunk.wav', audio, samplerate=sr)

    #TODO: also compute mfccs and other features with essentia
    #delta_mfccs = librosa.feature.delta(librosa_mfccs)
    #delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)


    feature_vector = []
    feature_vector = np.append(feature_vector, mfccs_mean)
    #feature_vector = np.append(feature_vector, np.mean(librosa_mfccs[1:], axis=1))
    #feature_vector = np.append(feature_vector, np.mean(delta_mfccs, axis=1))
    #feature_vector = np.append(feature_vector, np.mean(delta_delta_mfccs, axis=1))
    feature_vector = np.append(feature_vector, 1000*np.mean(librosa.feature.zero_crossing_rate(audio), axis=1))
    # feature_vector = np.append(feature_vector, np.mean(librosa.feature.spectral_centroid(audio, sr=sr), axis=1))
    # feature_vector = np.append(feature_vector, np.mean(librosa.feature.spectral_rolloff(audio, sr=sr), axis=1))
    # feature_vector = np.append(feature_vector, np.mean(librosa.feature.spectral_bandwidth(audio, sr=sr), axis=1))
    # feature_vector = np.append(feature_vector, np.mean(librosa.feature.spectral_flatness(audio), axis=1))

    return feature_vector


def get_song_info(doc, dataset):
    source_filenames = []
    instr_fams = []
    instr_classes = []
    instr_subclasses = []
    stems = doc["stems"]

    if dataset == 'slakh':
        song_name = doc["audio_dir"].split('/')[0]
        for stem in stems:
            if stems[stem]["audio_rendered"]:
                source_filenames.append(stem + ".wav")
                instr_fams.append(stems[stem]["++FAMILY++"])
                instr_classes.append(stems[stem]["++CLASS++"])
                instr_subclasses.append(stems[stem]["++SUBCLASS++"])
    else:
        song_name = doc["mix_filename"][0:-8]
        for stem in stems:
            raw_files = stems[stem]["raw"]

            for raw_file in raw_files:
                source_filenames.append(raw_files[raw_file]["filename"])
                instr_fams.append(raw_files[raw_file]["++FAMILY++"])
                instr_classes.append(raw_files[raw_file]["++CLASS++"])
                instr_subclasses.append(raw_files[raw_file]["++SUBCLASS++"])

    return song_name, source_filenames, instr_fams, instr_classes, instr_subclasses


def get_audio_chunk(wav_path):
    ''' returns chunk with the highest rms! 
    if all chunks are below a threshold --> returns None '''
    rms_threshold = 10 ** (-60 / 20)
    chunk_length = 4 * samplerate

    song_length = torchaudio.info(wav_path)[0].length
    hop = (song_length - chunk_length) // 8

    chunks = []
    rms = []
    for i in range(8):
        audio, _ = soundfile.read(wav_path, start=i*hop, frames=chunk_length, dtype='float32')
        chunks.append(audio)
        rms.append(np.sqrt(np.mean(np.square(audio))))

    max_rms = max(rms)
    if max_rms < rms_threshold:
        return None

    max_index = rms.index(max_rms)
    return chunks[max_index]


def get_metadata_files(dataset, splits=[]):
    metadata_files = []
    for split in splits:
        metadata_files += glob(os.path.join("./../../data/splits", dataset, split, "*.yaml"))

    return metadata_files


def get_wav_path(dataset, song_name, source):
    if dataset == 'slakh':
        wav_path = os.path.join(audiodata_path, 'slakh', "wav", song_name, "stems", source)
    elif dataset == 'mixing-secrets':
        wav_path = os.path.join(audiodata_path, 'mixing-secrets', "wav", song_name + '_Full', source)
    elif dataset == 'medleydb':
        wav_path = os.path.join(audiodata_path, 'medleydb', "wav", song_name, "RAW", source)
    return wav_path


def loudness_normalization(audio):
    meter = pyln.Meter(samplerate)
    loudness = meter.integrated_loudness(audio)
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, -30.0)
    return loudness_normalized_audio


def peak_normalize(audio):
    return audio / np.max(np.abs(audio))


def make_data(use_instr_classes=True, instr_fam='all', datasets=[], splits='train', max_sources_per_dataset=2000):
    x = []
    labels = []
    for dataset in datasets:
        source_count = 0
        metadata_files = get_metadata_files(dataset, splits=splits)
        for file in tqdm.tqdm(metadata_files):
            if source_count > max_sources_per_dataset:
                break

            with open(file, 'r') as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)

            song_name, source_filenames, instr_fams, instr_classes, instr_subclasses = get_song_info(doc, dataset)
            for i in range(len(source_filenames)):
                if instr_fam == 'all' or instr_fam == instr_fams[i]:
                    wav_path = get_wav_path(dataset, song_name, source=source_filenames[i])

                    audio = get_audio_chunk(wav_path)  # don't load whole song !!!

                    if audio is not None:
                        if use_instr_classes and instr_classes[i] != 'unknown' or not use_instr_classes:
                            source_count += 1
                            #audio = loudness_normalization(audio)
                            audio = peak_normalize(audio)
                            #TODO: listen to audio chunks
                            features = compute_timbre_features(audio, sr=samplerate, n_mfccs=12)
                            x.append(features)
                            if use_instr_classes:
                                labels.append(instr_classes[i])
                            else:
                                labels.append(instr_fams[i])

    # standardization: x_standard = (x - mu) / sigma
    #x = preprocessing.scale(x, axis=0) # is this necessary???
    mean1 = np.mean(x, axis=0)
    mean2 = np.mean(x, axis=1)
    std1 = np.std(x, axis=0)
    std2 = np.std(x, axis=1)

    return x, labels

def main():
    writer = SummaryWriter()
    x, labels = make_data(use_instr_classes=False, 
                        instr_fam='all',
                        datasets=['medleydb', 'mixing-secrets'],
                        splits=['valid', 'train', 'test'],
                        max_sources_per_dataset=50)
    
    x = np.array(x)
    #writer.add_embedding(x, metadata=labels)

    tsne = TSNE(n_components=2, verbose=1, random_state=13, perplexity=30) #try different random_states and perplexities
    z = tsne.fit_transform(x)

    x=z[:,0]
    y=z[:,1]
    #sns.scatterplot(x=x, y=y, hue=labels, ec=None)
    px.scatter(x, y)

    plt.savefig('t-sne_plot.png')


if __name__ == '__main__':
    main()