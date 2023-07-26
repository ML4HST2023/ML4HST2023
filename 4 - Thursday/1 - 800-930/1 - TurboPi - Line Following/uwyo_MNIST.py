# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:15:53 2023

@author: jblaney1
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader as dl

from uwyo_models import ann
import uwyo_common as common
from uwyo_trainer import trainer

__name__='uwyo_common'

image_width=28
image_height=28
input_shape = int(image_width*image_height)
batch_size = 16

mnist_linear = ann(name='test')
mnist_linear.create_model(model_type='linear', inputs=input_shape, outputs=10, activations=['relu','softmax'], linear_batch_normalization=True, linear_dropout=0.1, neurons=[32,16], 
                          cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, pooling=None, cnn_batch_normalization=None, cnn_dropout=None,
                          rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)

linear_transform = transforms.Compose([transforms.ToTensor()])

train_mnist = datasets.MNIST('/MNIST/', train=True, download=True, transform=linear_transform)
train_data_loader = dl(train_mnist,
                       batch_size=batch_size,
                       shuffle=True)

test_mnist = datasets.MNIST('/MNIST/', train=False, download=True, transform=linear_transform)
tests_data_loader = dl(test_mnist)

test_trainer = trainer(name='test',
                       model=mnist_linear.model,
                       loss='crossentropyloss',
                       optimizer='adam',
                       lr=0.01)

history = test_trainer.train(num_epochs=1, train=train_data_loader, reshape=(batch_size, input_shape))

accuracy = test_trainer.test(test=tests_data_loader, reshape=(1, input_shape))

test_trainer.plot_history(history)

common.Print(f"Model: {mnist_linear.report['name']} | Test Accuracy: {accuracy}")


mnist_cnn = ann(name='test')
mnist_cnn.create_model(model_type='cnn', inputs=1, outputs=10, activations=['relu','relu','softmax'], linear_batch_normalization=False, linear_dropout=0.1, neurons=[32,16], 
                         cnn_type='2d', channels=[32,16,8], image_width=image_width, image_height=image_height, kernels=(3,3), strides=None, paddings=None, pooling='maxpool2d', cnn_batch_normalization=True, cnn_dropout=0.1,
                         rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)

cnn_transform = transforms.Compose([transforms.Resize([image_width, image_height]),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])

train_mnist = datasets.MNIST('/MNIST/', train=True, download=True, transform=cnn_transform)
train_data_loader = dl(train_mnist,
                       batch_size=batch_size,
                       shuffle=True)

test_mnist = datasets.MNIST('/MNIST/', train=False, download=True, transform=cnn_transform)
tests_data_loader = dl(test_mnist)

test_trainer.model = mnist_cnn.model

history = test_trainer.train(num_epochs=10, train=train_data_loader)

accuracy = test_trainer.test(test=tests_data_loader)

test_trainer.plot_history(history)

common.Print(f"Model: {mnist_cnn.report['name']} | Test Accuracy: {accuracy}")