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
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler

#загружаем данные о классах птиц и их id в DataFrame
metadata = pd.read_csv('birdsong_metadata.csv', usecols = ['file_id', 'english_cname'])

#переводим имена классов птиц в int от 0 до 87 (всего 88 вилов птиц)
lb_make = LabelEncoder()
names = metadata.english_cname.unique()
label_int = lb_make.fit_transform(names)

#создаём словарь где ключ это наименование птицы, а значение - номер её класса
catigories = {}
for i in range(len(label_int)):
    catigories.update({names[i]: label_int[i]})


#определяем функцию которая обрезает слишком длинные аудиофайлы до 10 секунд
#если длительность аудио меньше 10 сек, то слева и справа добавляются сэмплы ввиде нулей так,
#что общая длительность аудио становится 10 сек
def cut_sound(path, sound_length = 10):
        sample, sample_rate = librosa.load(path, sr=None)

        #конкретное место аудио, от куда взять 10 сек определяется случайно
        if sample.shape[0]/sample_rate > sound_length*2:
            start_point = np.random.randint(1,10)
            sample = sample[sample_rate*start_point:sample_rate*start_point + sound_length*sample_rate]

        elif sample.shape[0]/sample_rate > sound_length:
            sample = sample[:sample_rate*sound_length]

        elif sample.shape[0]/sample_rate < sound_length:
            sample_start = random.randint(0, sound_length*sample_rate - sample.shape[0])
            sample_end = sound_length*sample_rate - int(sample.shape[0]) - sample_start
            sample = np.concatenate([np.zeros(sample_start), sample, 
                                np.zeros(sample_end)])

        return sample, sample_rate

#полученные сэмплы в случайном порядке изменяются
def augment(sample, sample_rate):
    #изменяем высоту звука
    pitch = np.random.uniform(0.5,8)
    sample_pitch = librosa.effects.pitch_shift(sample, sample_rate, 
                                               n_steps = pitch)
    
    #двигаем значения по оси времени
    sec_len = len(sample_pitch)/sample_rate
    shift = np.random.randint(1, sec_len-1)
    sample = np.roll(sample_pitch, shift*sample_rate)

    #добавляем шум
    choice = np.random.randint(0,2)
    if choice == 1:
      noise = np.random.randn(len(sample))
      sample = sample + 0.002*noise

    return sample



#строим мел спектограмму и случайным образом доавляем маски по оси времени и частоты звука
#ввиде средних значений по всему сэмплу
def mel_spec(signal, sample_rate, n_fft, hop_length, n_mels):
    mel_gram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,
                                    fmax=8000, hop_length = hop_length, n_fft = n_fft)
    mel_gram__dB = librosa.power_to_db(mel_gram)
    mel_gram__dB = mel_gram__dB[np.newaxis, ...]

    #опредеяем среднее значение по всему сэмплу
    spec_mean = mel_gram__dB[0].mean()

    #вырезаем случайное место по оси времени и вставляем средние значения по сэмплу
    height = mel_gram__dB.shape[1]
    width = np.random.randint(10, 60)
    time_cut = np.full((height,width), spec_mean)
    start_time = np.random.randint(1, mel_gram__dB.shape[2]-60)
    mel_gram__dB[0][:,start_time:start_time+width] = time_cut

    #вырезаем случайное место по оси частоты и вставляем средние значения по сэмплу
    width = mel_gram__dB.shape[2]
    height = np.random.randint(5, 10)
    freq_cut = np.full((height,width), spec_mean)
    start_freq =  np.random.randint(1, mel_gram__dB.shape[1]-10)
    mel_gram__dB[0][start_freq:start_freq+height,:] = freq_cut

    return mel_gram__dB

#создаём датасет, который пропускает сэмплы через все вышеописанные функции
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
        signal = augment(sample = signal, sample_rate = sample_rate)
        spec = mel_spec(signal = signal, sample_rate = sample_rate, n_fft = 1024, 
                 hop_length = 512, n_mels = 64)

        label = self.metadata.loc[idx, 'english_cname']
        label = self.str2lab[label]

        return spec, label


data_sound = birdsound(metadata = metadata, str2lab = catigories)

#берём 90% данных для обучения и 10% для проверки
total_items = len(data_sound)
num_train = round(total_items * 0.9)
num_val = total_items - num_train
train_data, val_data = random_split(data_sound, [num_train, num_val])

#создаём даталоадер с батчем 16 для обучения и 8 для контроля
train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)
