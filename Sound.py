import librosa
import random 
import numpy as np 
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import os
import torch 
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from torchvision import models
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler


metadata = pd.read_csv('birdsong_metadata.csv', usecols = ['file_id', 'english_cname'])
lb_make = LabelEncoder()
names = metadata.english_cname.unique()
label_int = lb_make.fit_transform(names)
catigories = {}
for i in range(len(label_int)):
    catigories.update({names[i]: label_int[i]})



def cut_sound(path, sound_length = 10):
        sample, sample_rate = librosa.load(path, sr=None)

        if sample.shape[0]/sample_rate > sound_length*2:
            sample = sample[sample_rate*sound_length:sample_rate*sound_length*2]

        elif sample.shape[0]/sample_rate > sound_length:
            sample = sample[:sample_rate*sound_length]

        elif sample.shape[0]/sample_rate < sound_length:
            sample_start = random.randint(0, sound_length*sample_rate - sample.shape[0])
            sample_end = sound_length*sample_rate - int(sample.shape[0]) - sample_start
            sample = np.concatenate([np.zeros(sample_start), sample, 
                                np.zeros(sample_end)])

        return sample, sample_rate


def mel_spec(signal, sample_rate, n_fft, hop_length, n_mels):
    mel_gram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,
                                    fmax=8000, hop_length = hop_length, n_fft = n_fft)
    mel_gram__dB = librosa.power_to_db(mel_gram)
    mel_gram__dB = mel_gram__dB[np.newaxis, ...]

    return mel_gram__dB



class birdsound(Dataset):
    def __init__(self, metadata, str2lab):
        self.metadata = metadata
        self.str2lab = str2lab

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        id = self.metadata.loc[idx, 'file_id']
        path_to_sound = 'songs\\songs\\xc'+ str(id) + '.flac'
        signal, sample_rate = cut_sound(path = path_to_sound)
        spec = mel_spec(signal = signal, sample_rate = sample_rate, n_fft = 1024, 
                 hop_length = 512, n_mels = 64)

        label = self.metadata.loc[idx, 'english_cname']
        label = self.str2lab[label]

        return spec, label



data_sound = birdsound(metadata = metadata, str2lab = catigories)

total_items = len(data_sound)
num_train = round(total_items * 0.8)
num_val = total_items - num_train
train_data, val_data = random_split(data_sound, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)


 


model = models.resnet18(pretrained=False)
model.fc.out_features = 88
model.conv1 =  nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(torch.float64) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

class AudioClassifier():
    def __init__(self, model, train_data, val_data, epochs, lr_rate = 0.001):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.lr_rate = lr_rate
        self.epochs = epochs

    def train(self):
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr = self.lr_rate)
        lr_step = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(self.epochs):
            print(f'Обучение эпохи: {epoch}')
            batch_counter = 0
            accuracy_train = 0

            for image, label in self.train_data:
                batch_counter += 1
                image = image.to(device = device, dtype=torch.float64)
                label = label.to(device)
                image.requires_grad_()
                optimizer.zero_grad()
                self.model.train()

                output = self.model(image)
                error = loss(output, label)
                error.backward()
                optimizer.step()
                _, pred = torch.max(output, dim = 1)
                accuracy_train += accuracy_score(label, pred)
            
            print('Training accuracy = ', accuracy_train/batch_counter)

            with torch.no_grad():
                accuracy_val = 0
                batch_counter = 0
                for image, label in self.val_data:
                    batch_counter +=1
                    image = image.to(device)
                    label = label.to(device)
                    self.model.eval()

                    output = self.model(image)

                    #Получаем значения предсказаний в формате integer для использования в метрике
                    _, pred = torch.max(output, dim = 1)
                    accuracy_val += accuracy_score(label, pred)

                lr_step.step()
                print('Validation accuracy = ', accuracy_val/batch_counter)

 

net = AudioClassifier(model = model, train_data = train_dl, val_data = val_dl, 
                      epochs = 15, lr_rate = 0.001)

net.train()




