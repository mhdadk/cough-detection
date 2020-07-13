import os
from glob import glob
import librosa
import torch
import numpy as np
from net import Net
import pickle

data_dir = '../../data_audio'
folders = os.listdir(data_dir)

# data to be extracted

X = np.empty((0,1024))
labels = []

# initialize net

param_path = 'mx-h64-1024_0d3-1.17.pkl'
net = Net(param_path)
net.eval()

for label,folder in enumerate(folders): # classes
    for file in os.listdir(os.path.join(data_dir,folder)):
        
        # load the WAV file
        
        y,sr = librosa.load(path = os.path.join(data_dir,folder,file),
                            sr = None)
        
        # convert to mono if necessary
        
        if y.ndim > 1:
            y = librosa.to_mono(y = y)
            
        # resample to 44.1 kHz if necessary
        
        target_sr = 44100
        
        if sr != target_sr:
            y = librosa.resample(y = y,
                                 orig_sr = sr,
                                 target_sr = target_sr)
            
        # if signal is shorter than 3 ms x 44100 Hz = 128 samples, then pad it
        # with zeros. This is necessary because PyTorch throws an error if
        # one of the image dimensions is less than 128
        
        if y.shape[0] < 128:
            
            pad_size = 128 - y.shape[0]
            
            y = np.pad(y,(0,pad_size))
        
        # compute the Mel-scale spectrogram
        
        mel_specgram = librosa.feature.melspectrogram(y = y,
                                                      sr = target_sr,
                                                      n_fft = 1024,
                                                      hop_length = 512,
                                                      n_mels = 128)
        
        # apply log transformation to dilate values
        
        log_mel_specgram = librosa.power_to_db(mel_specgram)
        
        # Add batch and channel dimensions because input to net needs to be
        # 4D (N x C x H x W)
        
        target_shape = (1,1,log_mel_specgram.shape[0],log_mel_specgram.shape[1])
        log_mel_specgram = np.reshape(log_mel_specgram,target_shape)
        
        # convert to torch tensor
        
        log_mel_specgram = torch.from_numpy(log_mel_specgram)
        
        # extract feature vector
        
        with torch.no_grad():
            fv = net(log_mel_specgram).numpy()
        
        # append to dataset
        
        X = np.vstack((X,fv))
        labels.append(label)    

with open('X.npy','wb') as f:
    np.save(file = f, arr = X)

with open('labels.pkl','wb') as f:
    pickle.dump(labels,f)