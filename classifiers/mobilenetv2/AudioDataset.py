import torch

from torch import nn

from glob import glob

import random

import librosa

import torchaudio

import torchvision as tv

from sklearn.model_selection import train_test_split

import scipy

import numpy as np

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,data_dir,sample_rate,mode):
        
        self.sample_rate = sample_rate
                
        # paths to all cough and non-cough files
        
        self.paths_all = glob(data_dir+'/*/*.wav')
        
        # make sure cough and non-cough files are mixed
        
        random.shuffle(self.paths_all)
        
        # create list of labels
        
        self.labels_all = []
        
        for path in self.paths_all:
            if 'not_coughs' in path:
                self.labels_all.append(0)
            else:
                self.labels_all.append(1)
        
        # seed for random number generator
        
        rng_seed = 42
        
        # create a 70-30 training-testing split
        
        (self.paths_train,
         self.paths_test,
         self.labels_train,
         self.labels_test) = train_test_split(self.paths_all,
                                              self.labels_all,
                                              train_size = 0.7,
                                              test_size = 0.3,
                                              random_state = rng_seed,
                                              shuffle = True,
                                              stratify = self.labels_all)
        
        # create a 70-15-15 training-validation-testing split
                                              
        (self.paths_val,
         self.paths_test,
         self.labels_val,
         self.labels_test) = train_test_split(self.paths_test,
                                              self.labels_test,
                                              train_size = 0.5,
                                              test_size = 0.5,
                                              random_state = rng_seed,
                                              shuffle = True,
                                              stratify = self.labels_test)
                                              
        # pick the training, validation, or testing set based on the mode
        
        if mode == 'train':
            self.paths = self.paths_train
            self.labels = self.labels_train
        elif mode == 'val':
            self.paths = self.paths_val
            self.labels = self.labels_val
        else:
            self.paths = self.paths_test
            self.labels = self.labels_test
    
    def __len__(self):
        
        return len(self.paths)
    
    def __getitem__(self,idx):
        
        # assign labels for cough and non-cough files
        
        label = torch.tensor([self.labels[idx]])
        
        # load waveform
                
        waveform,_ = librosa.load(path = self.paths[idx],
                                  sr = self.sample_rate,
                                  mono = True)
        
        # if the audio file is shorter than 9 seconds, pad it with zeros to be
        # 10 seconds long, and if it is longer than 11 seconds, decimate it to
        # be 10 seconds long
        
        if (waveform.shape[0] < 9*self.sample_rate):
            
            pad_size = (10*self.sample_rate) - waveform.shape[0]
            
            waveform = np.pad(waveform,(0,pad_size))
        
        elif (waveform.shape[0] > 11*self.sample_rate):
            
            waveform = scipy.signal.resample(waveform,
                                             10*self.sample_rate,
                                             window = 'hann')
        
        # to put tensors on GPU if available
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # convert to tensor from numpy array
        
        waveform = torch.tensor(waveform).to(device)
        
        # compute spectrogram using short-time Fourier transform
        
        spec_func = torchaudio.transforms.Spectrogram(
                    n_fft = 1024,
                    hop_length = 56,
                    power = 2.0,
                    normalized = True)
        
        # convert frequencies to Mel scale
        
        mel_func = torchaudio.transforms.MelScale(
                   n_mels = 224,
                   sample_rate = self.sample_rate)
        
        # dilate small values
        
        log_func = torchaudio.transforms.AmplitudeToDB(stype = 'power')
        
        log_mel_spec = log_func(mel_func(spec_func(waveform)))

        # convert to RGB image
        
        log_mel_spec = torch.unsqueeze(log_mel_spec,dim=0)
        
        log_mel_spec = torch.cat(3*[log_mel_spec],dim=0)
        
        # downsample image to fit input to network. nn.functional.interpolate()
        # expects B x C x H x W dimensions for input image.
        
        log_mel_spec = torch.unsqueeze(log_mel_spec,dim=0)
        
        log_mel_spec = nn.functional.interpolate(
                       log_mel_spec,
                       size = (224,224),
                       mode = 'bilinear',
                       align_corners = False)
        
        log_mel_spec = torch.squeeze(log_mel_spec,dim=0)
        
        # standardization
        
        normalize_func = tv.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
        
        log_mel_spec = normalize_func(log_mel_spec)
        
        return (log_mel_spec,label)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    from playsound import playsound
    
    data_dir = '../../data'
    
    sample_rate = 22050 # Hz
    
    mode = 'test'
    
    dataset = AudioDataset(data_dir,sample_rate,mode)
    
    idx = random.randint(0,len(dataset))
    
    image,label = dataset[idx]
    
    print('\nShowing spectrogram for:\n{}'.format(dataset.paths[idx]))
    
    plt.imshow(image[0])
        
    playsound(dataset.paths[idx])
