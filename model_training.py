import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
import librosa
import random 
import numpy as np 
import pandas as pd
import os
import torch 
from torch.utils.data import DataLoader, Dataset, random_split
#импортируем даталоадеры и словарь категорий видов птиц
from data_processing import catigories, train_dl, val_dl 

#загружаем архитектуру ResNet 18 (заранее не обученную)
model = models.resnet18(pretrained=False)

#в выходном слоя меняем 1000 выходных нейронов на 88 т.к. у нас 88 классов
model.fc.out_features = 88

#заменяем глубину входных данных в первый свёрточный слой с 3 на 1, т.к. 
#наши данные имеют размерность 1 в глубину
model.conv1 =  nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#переводим все значения в модели в формат float64 т.к. входные данные в таком формате
model = model.to(torch.float64) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#определяем класс, который включает функцию обучения нейронной сети.
#Функция потерь - CrossEntropyLoss
#Способ оптимизации - Adam
#Скорость обучения замедляется в 2 раза каждые 20 эпох
acc_tr_final = []
acc_val_final = []
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
        lr_step = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        #стадия обучения
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
                accuracy_train += accuracy_score(label.cpu(), pred.cpu())
            
            print('Training accuracy = ', accuracy_train/batch_counter)
            acc_tr_final.append(accuracy_train/batch_counter)

            #стадия валидации
            with torch.no_grad():
                accuracy_val = 0
                batch_counter = 0
                best_auc = 0
                for image, label in self.val_data:
                    batch_counter +=1
                    image = image.to(device = device, dtype=torch.float64)
                    label = label.to(device)
                    self.model.eval()

                    output = self.model(image)

                    #Получаем значения предсказаний в формате integer для использования в метрике
                    _, pred = torch.max(output, dim = 1)
                    accuracy_val += accuracy_score(label.cpu(), pred.cpu())

                print('Validation accuracy = ', accuracy_val/batch_counter)
                acc_val_final.append(accuracy_val/batch_counter)

                #если точность в этой эпохе лучше, то сохраняем веса
                if accuracy_val/batch_counter > best_auc:
                  torch.save(model.state_dict(), 'songs//songs//best_weight.pth')

            lr_step.step()

        
#загружаем класс в net и обучаем
net = AudioClassifier(model = model, train_data = train_dl, val_data = val_dl, 
                      epochs = 40, lr_rate = 0.001)
#net.train()
#поскольку модель была обучена, можно просто загрузить получившиеся веса
model.load_state_dict(torch.load('best_weight.pth', map_location=device))

