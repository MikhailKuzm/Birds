import librosa
import random 
import numpy as np 
import pandas as pd
import os
import torch 
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score

#загружаем модель, словарь с категориями птиц и валидационный dataloader
from model_training import model
from data_processing import catigories,  val_dl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#определяем функцию predict, которая принимает одну мел-спектограмму из контрольного
#набора данных и возвращает предказание
def predict(image, dict4pred = catigories, model = model):
    #если входные данные не тензор, переводим в тензор
    if not torch.is_tensor(image):
        image = torch.tensor(x, dtype = torch.float64)
    #если необходимо, добавляем размерность, потому что модель ожидает
    #что входные данные подаются батчами
    if len(image.shape)<3:
        image = image.unsqueeze(0)
    if len(image.shape)<4:
        image = image.unsqueeze(0)

    image = image.to(device = device, dtype=torch.float64)
    model.eval()
    
    with torch.no_grad():
        prediction = model(image)
        #получаем предсказание ввиде int
        prediction = torch.max(prediction, dim = 1)[1]

        #находим, какой ключ в словаре видов птиц соответствует полученному prediction
        cat_keys = list(catigories.keys())
        cat_value = list(catigories.values())
        pred_pos = cat_value.index(prediction)
        prediction = cat_keys[pred_pos]

        return prediction


#для проверки можно взять один батч из валидационного набора данных и предсказать его классы
val = next(iter(val_dl))
y_true = []
pred = []

#предсказываем вид птицы по данным из валидационного набора данных
for i in range(len(val[1])):
    pred.append(predict(val[0][i]))
    cat_keys = list(catigories.keys())
    cat_value = list(catigories.values())
    pred_pos = cat_value.index(val[1][i])
    y_true.append(cat_keys[pred_pos])

print(y_true, '\n',
      pred)
accuracy_score(y_true, pred)

