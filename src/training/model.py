# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class BackboneCNN(nn.Module):
    '''
    Backbone network for feature extraction
    '''
    def __init__(self, samplerate, model_config):
        super(BackboneCNN, self).__init__()
        if model_config is None:
            n_channels = 128
            n_fft = 512
            n_mels = 128
        else:
            n_channels = model_config.n_channels
            n_fft = model_config.n_fft
            n_mels = model_config.n_mels

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=samplerate,
                                                         n_fft=n_fft,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels)
        self.layer2 = Conv_2d(n_channels, n_channels)
        self.layer3 = Conv_2d(n_channels, n_channels*2)
        self.layer4 = Conv_2d(n_channels*2, n_channels*2)
        self.layer5 = Conv_2d(n_channels*2, n_channels*2)
        self.layer6 = Conv_2d(n_channels*2, n_channels*2)
        self.layer7 = Conv_2d(n_channels*2, n_channels*4)


    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # after 7 layers with 2x2 max pooling the freq dimension is 1 when using 128 mel bins
        x = x.squeeze(2)

        # max pooling over time dimension
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        return x  # output tensor size --> [batchsize, 512]


class ClassPredictor(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self, samplerate, n_classes, model_config=None):
        super(ClassPredictor, self).__init__()
        if model_config is None:
            n_channels = 128
        else:
            n_channels = model_config.n_channels
        
        self.backbone = BackboneCNN(samplerate, model_config)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):        
        x = self.backbone(x)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, p, dim):
        super(L2Norm, self).__init__()
        self.normalize = F.normalize
        self.p = p
        self.dim = dim
        
    def forward(self, x):
        x = self.normalize(x, p=self.p, dim=self.dim)
        return x


class TimbrePredictor(nn.Module):
    '''
    Predicts timbre features such as MFCCs and loudness for a specific instrument class
    '''
    def __init__(self, samplerate, size_of_feature_vectors, model_config=None):
        super(TimbrePredictor, self).__init__()
        if model_config is None:
            n_channels = 128
        else:
            n_channels = model_config.n_channels

        self.backbone = BackboneCNN(samplerate, model_config)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(n_channels*4, size_of_feature_vectors)    

    def forward(self, x):        
        x = self.backbone(x)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
