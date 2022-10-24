"""
predict the instruments in your own music files
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import argparse
import torch
import pyloudnorm as pyln
from sklearn.preprocessing import LabelBinarizer

from model import ClassPredictor
import utils


class Predictor:
    def __init__(self, model, model_load_path, path_to_audiofile, samplerate, model_input_length, loudn_threshold):
        self.model = model
        self.model_load_path = model_load_path
        self.samplerate = samplerate
        self.input_length = model_input_length * samplerate
        self.chunks_per_track = 16
        self.path_to_audiofile = path_to_audiofile
        self.loudn_threshold = loudn_threshold
        self.loudn_meter = pyln.Meter(self.samplerate)

        # cuda
        num_of_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_of_gpus}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using Device: {self.device}")

        self.build_model()
        
        instr_families = utils.load_classes(type='instr_families')
        instr_classes = utils.load_classes(type='instr_classes')
        self.mlb = LabelBinarizer().fit(instr_families + instr_classes)


    def build_model(self):
        self.load(self.model_load_path)
        self.model.to(self.device)
        self.model = self.model.eval()


    def load(self, filename):
        state_dict = torch.load(filename)
        if 'backbone.spec.mel_scale.fb' in state_dict.keys():
            self.model.backbone.spec.mel_scale.fb = state_dict['backbone.spec.mel_scale.fb']
        self.model.load_state_dict(state_dict)

    
    def is_silent(self, audio):
        """ returns True if audio is silent (below threshold) """
        loudn = self.loudn_meter.integrated_loudness(audio)
        if loudn > self.loudn_threshold:
            return False
        return True


    def construct_batch(self):
        audio, sr = librosa.load(path=self.path_to_audiofile, sr=self.samplerate, mono=True)
        #audio, sr = soundfile.read(self.path_to_audiofile, dtype='float32')
        song_length = len(audio)
        hop = (song_length - self.input_length) // self.chunks_per_track
        audio_batch = np.ndarray(0)
        batchsize = 0
        for i in range(self.chunks_per_track):
            chunk = audio[i*hop:i*hop+self.input_length]
            if not self.is_silent(chunk):
                batchsize += 1
                chunk, _ = utils.peak_normalize_if_clipping(chunk)
                audio_batch = np.vstack((audio_batch, chunk)) if batchsize > 1 else chunk

        return torch.from_numpy(audio_batch)


    def predict(self):
        with torch.no_grad():
            x = self.construct_batch()
            x = x.to(self.device, dtype=torch.float32)
            out = self.model(x)

            # average over all chunks in the batch
            out = out.detach().cpu().numpy()
            out_average = out.mean(axis=0)
            self.create_barplot(out_average)


    def create_barplot(self, estim_prob):
        result = dict(zip(self.mlb.classes_, estim_prob.tolist()))
        result_list = list(sorted(result.items(), key=lambda x: x[1]))
        fig = plt.figure(figsize=[5, 10])
        plt.barh(np.arange(len(result_list)), [r[1] for r in result_list], align="center")
        plt.yticks(np.arange(len(result_list)), [r[0] for r in result_list])
        plt.tight_layout()
        audio_fn = self.path_to_audiofile.split('/')[-1]
        fig_fn = audio_fn.split('.')[0] + '.png'
        plt.savefig(fig_fn)
        plt.show()



def main(config):
    model = ClassPredictor(samplerate=32000,
                            n_classes=15)

    predictor = Predictor(model, 
                        config.model_load_path, 
                        config.path_to_audiofile, 
                        samplerate=32000, 
                        model_input_length=4,
                        loudn_threshold=-40)
    predictor.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path_to_audiofile', type=str, default='.')
    parser.add_argument('--model_load_path', type=str, default='./../../models/classifier-loudn_threshold=-40/best_model.pth')

    config = parser.parse_args()

    print(config)
    main(config)
