# -*- coding: utf-8 -*-
"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/01/2023

Purpose:
Used as a template file for creating pytorch ML models and training them. Currently
the dataloading is only implemented for images.

Functions:
 - 

Included with: 
 - uwyo_common.py
 - uwyo_crop.py 
 - uwyo_MNIST.py
 - uwyo_models.py
 - uwyo_preprocessor.py
 - uwyo_train_detector.py (current file)
 - uwyo_trainer.py
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader as dl

from uwyo_models import ann
import uwyo_trainer as trainer

def mean_std(loader):
    images, labels = next(iter(loader))
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std

width = 40
height = 30
model_creator = ann(name='detector')
model_creator.create_model(model_type='cnn', inputs=1, outputs=1, neurons=[4], activations=['relu', 'relu', 'sigmoid'], linear_batch_normalization=False, linear_dropout=None,
                           cnn_type='2d', channels=[8], image_width=width, image_height=height, kernels=(11,11), strides=None, paddings=None, pooling='maxpool2d', cnn_batch_normalization=True, cnn_dropout=0.1,
                           rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)

cnn_model = model_creator.model

dataset = 'tape'
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.ToTensor()])
loader = dl(datasets.ImageFolder(f'../TurboPi/Data/{dataset}/training/', transform=transform), batch_size=245)

mean, std = mean_std(loader)

normalize = transforms.Normalize(mean, std)

print(f'Mean : {mean} | STD : {std}')

train_transform = transforms.Compose([transforms.Resize([height,width]),
                                      transforms.Grayscale(),
                                      transforms.ColorJitter(),
                                      transforms.RandomPerspective(),
                                      transforms.ToTensor(),
                                      normalize])

tests_transform = transforms.Compose([transforms.Resize([height,width]),
                                      transforms.Grayscale(),
                                      transforms.ToTensor(),
                                      normalize])

batch_size = 64


train = dl(datasets.ImageFolder(f'../TurboPi/Data/{dataset}/training/', transform=train_transform), batch_size, shuffle=True)
valid = dl(datasets.ImageFolder(f'../TurboPi/Data/{dataset}/validation/', transform=train_transform), batch_size, shuffle=False)
tests = dl(datasets.ImageFolder(f'../TurboPi/Data/{dataset}/testing/', transform=tests_transform), batch_size, shuffle=False)

history, cnn_model = trainer.train(cnn_model, 100, train, valid, thresh=0.95)

acc = trainer.test(cnn_model, tests, verbose=1, thresh=0.95)

trainer.plot_history(history)

model_script = torch.jit.script(cnn_model)
model_script.save('line_follower.pt')