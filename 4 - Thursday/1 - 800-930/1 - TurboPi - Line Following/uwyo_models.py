# -*- coding: utf-8 -*-
"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/01/2023

Purpose:
To provide streamlined access to the python api by abstracting the model
building and creating an interface to use when passing the model into a 
training module. 

Functions:
 model
 - __init__(name, model=None)
 - attribute_get(name)
 - attribute_set(name, value)
 - build_cnn()
 - build_linear()
 - build_rnn()
 - create_model(model_type, inputs, outputs, neurons, activations, linear_batch_normalization=None, linear_dropout=None,
                cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, cnn_batch_normalization=None, pooling=None,
                rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)
 - expand_activations(activations, length)
 - process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, cnn_batch_normalization, pooling)
 - process_global_inputs(inputs, outputs, neurons, activations, batch_normalization, dropout)
 - process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
 - set_activation(name)
 - set_pooling(name)

Included with: 
 - common.py 
 - model.py (current file)
 - trainer.py 
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

# Outside Dependencies
import torch
from torch import nn

from torchsummary import summary

# In House Dependencies
import uwyo_common as common

__name__ = 'uwyo_models'


"""
    model(name, model=None)
    A class to store a pytorch model and to enable easy model creation using 
    lists instead of discrete function calls in your code.
"""
class ann():
    def __init__(self, name):
        super().__init__()
        self.report = {}
        self.history = {}
        self.model = None
        
        self.attribute_set('name', name)
        self.cnn_types = [None, '1d', '2d']
        self.rnn_types = [None, 'rnn', 'lstm', 'gru']
        self.model_types = ['linear', 'cnn', 'rnn']
        
    
    """
        attribute_get(name)
        A function to retrieve values stored in the report
        
        inputs:
            - name (string): The key to retrieve the information from
        outputs:
            -
    """
    def attribute_get(self, name):
        try:
            return self.report[name]
        except Exception as e:
            common.Print_Error('Model -> attribute get', e)
    
    
    """
        attribute_set(name, value)
        A function to update the report entry specified by name with value.
        
        inputs:
            - name (string): The entry to update
            - value (): The new value to store at the entry
        outputs:
            -
    """
    def attribute_set(self, name, value):
        try:
            self.report[name] = value
        except Exception as e:
            common.Print_Error('Model -> attribute set', e)
    
    
    """
        build_cnn()
        A function to build a cnn model based on the infomration stored in the 
        report dictionary and stores the resultant model as the model of this 
        object.
    """
    def build_cnn(self, ):
        
        common.Print('[INFO] Starting Build CNN Process')
        try:
            self.model = nn.Sequential()
            
            outputs = self.report['inputs']
            conv = nn.Conv1d if self.report['cnn type'] == '1d' else nn.Conv2d
            self.report['cnn batch norm function'] = nn.BatchNorm1d if self.report['cnn type'] == '1d' else nn.BatchNorm2d
            dummy = torch.ones((1, outputs, self.report['image width'], self.report['image height']))

            for i in range(0, self.report['cnn layers']):
                inputs = outputs
                outputs = self.report['channels'][i]
                self.model.add_module(f'conv {i}', 
                                      conv(in_channels=inputs,
                                           out_channels=outputs,
                                           kernel_size=self.report['kernels'][i],
                                           stride=self.report['strides'][i],
                                           padding=self.report['paddings'][i]))

                self.model.add_module(f'activation {i}', self.report['activations'][i]()) 

                outputs = self.model(dummy).shape[1]
                
                if self.report['pooling'] is not None:
                    self.model.add_module(f'pool {i}', self.report['pooling'](7,7))

                if self.report['cnn batch normalization']:
                    self.model.add_module(f'batch normalization {i}', self.report['cnn batch norm function'](num_features=outputs))

                if self.report['cnn dropout'] is not None:
                    if self.report['cnn dropout'][i] > 0:
                        self.model.add_module(f'dropout {i}', nn.Dropout(self.report['cnn dropout'][i]))
                
                common.Print_Status('Add CNN', i, self.report['cnn layers'])
                    
            self.model.add_module('flatten', nn.Flatten())
            outputs = self.model(dummy).shape[1]
            
            for j in range(0, self.report['linear layers']):
                inputs = outputs
                outputs = self.report['neurons'][j]
                self.model.add_module(f'linear {j}', nn.Linear(inputs, outputs))

                self.model.add_module(f'activation {j}', self.report['activations'][j+i]())   

                if self.report['linear batch normalization'] and not self.report['cnn batch normalization']:
                    self.model.add_module(f'batch normalization {j}', nn.BatchNorm1d(num_features=outputs))

                if self.report['linear dropout'] is not None:
                    if self.report['linear dropout'][j] > 0:
                        self.model.add_module(f'dropout {j}', nn.Dropout(self.report['linear dropout'][j]))

                common.Print_Status('Add Linear Layers', i, self.report['linear layers'])
            
            inputs = outputs
            outputs = self.report['outputs']
            self.model.add_module(f'linear {j+i+1}', nn.Linear(inputs, outputs))
            self.model.add_module(f'activation {j+i+1}', self.report['activations'][-1]())
            
        except Exception as e:
            common.Print_Error('Model -> build cnn', e)
            return
        
    
    """
        build_linear()
        A function to build a linear model based on the infomration stored in 
        the report dictionary and stores the resultant model as the model of 
        this object.
    """
    def build_linear(self, ):
        
        common.Print('[INFO] Starting Build Linear Process')
        try:
        
            self.model = nn.Sequential()
            
            outputs = self.report['inputs']
            
            for i in range(0, self.report['linear layers']):
                inputs = outputs
                outputs = self.report['neurons'][i]
                self.model.add_module(f'linear {i}', nn.Linear(inputs, outputs))
                self.model.add_module(f'activation {i}', self.report['activations'][i]())   
                
                if self.report['linear batch normalization']:
                    self.model.add_module(f'batch normalization {i}', nn.BatchNorm1d(num_features=outputs))
                
                if self.report['linear dropout'] is not None:
                    if self.report['linear dropout'][i] > 0:
                        self.model.add_module(f'dropout {i}', nn.Dropout(self.report['linear dropout'][i]))

                common.Print_Status('Add Linear Layers', i, self.report['linear layers'])
            
            inputs = outputs
            outputs = self.report['outputs']
            self.model.add_module(f'linear {i+1}', nn.Linear(inputs, outputs))
            self.model.add_module(f'activation {i+1}', self.report['activations'][-1]())

        except Exception as e:
            common.Print_Error('Model -> build linear', e)
            return
    
    
    """
        build_rnn()
        A function to build an rnn model based on the information stored in 
        the report dictionary and stores the resultant model as the model of 
        this object.
    """
    def build_rnn(self, ):
        common.Print('[INFO] Starting Build RNN Process')
        try:
            self.model = nn.Sequential()
            
            dummy = torch.ones((1,1,self.report['inputs']))
            
            if self.report['rnn type'] == 'rnn':
                self.model.add_module('rnn', nn.RNN(input_size=self.report['inputs'],
                                                    hidden_size=self.report['hidden size'],
                                                    num_layers=self.report['rnn layers'],
                                                    nonlinearity=self.report['non-linearity'],
                                                    bias=self.report['bias'],
                                                    batch_first=self.report['batch first'],
                                                    dropout=self.report['rnn dropout'],
                                                    bidirectional=self.report['bidirectional']))
            elif self.report['rnn type'] == 'lstm':
                self.model.add_module('lstm', nn.LSTM(input_size=self.report['inputs'],
                                                      hidden_size=self.report['hidden size'],
                                                      num_layers=self.report['rnn layers'],
                                                      bias=self.report['bias'],
                                                      batch_first=self.report['batch first'],
                                                      dropout=self.report['rnn dropout'],
                                                      bidirectional=self.report['bidirectional'],
                                                      proj_size=self.report['projection size']))
            elif self.report['rnn type'] == 'gru':
                self.model.add_module('gru', nn.GRU(input_size=self.report['inputs'],
                                                    hidden_size=self.report['hidden size'],
                                                    num_layers=self.report['rnn layers'],
                                                    bias=self.report['bias'],
                                                    batch_first=self.report['batch first'],
                                                    dropout=self.report['rnn dropout'],
                                                    bidirectional=self.report['bidirectional']))
                
            self.model.add_module('flatten', nn.Flatten())
        
            outputs = self.model(dummy).shape[1]
                
            for i in range(0, self.report['linear layers']):
                inputs = outputs
                outputs = self.report['neurons'][i]
                self.model.add_module(f'linear {i-1}', nn.Linear(inputs, outputs))
                self.model.add_module(f'activation {i-1}', self.report['activations'][i]())   
                
                if self.report['linear batch normalization']:
                    self.model.add_module(f'batch normalization {i}', nn.BatchNorm1d(num_features=outputs))
                
                if self.report['linear dropout'] is not None:
                    if self.report['linear dropout'][i] > 0:
                        self.model.add_module(f'dropout {i}', nn.Dropout(self.report['linear dropout'][i]))

                common.Print_Status('Add Linear Layers', i, self.report['linear layers'])
            
            inputs = outputs
            outputs = self.report['outputs']
            self.model.add_module(f'linear {i+1}', nn.Linear(inputs, outputs))
            self.model.add_module(f'activation {i+1}', self.report['activations'][-1]())
            
        except Exception as e:
            common.Print_Error('Model -> build rnn', e)
            return
        
    
    """
        create_model(model_type, inputs, outputs, neurons, activations, linear_batch_normalization=None, linear_dropout=None,
                     cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, pooling=None, cnn_batch_normalization=None, cnn_dropout=None,
                     rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None)
        A function to automate the building of different network architectures.
        It can build linear, cnn, and rnn models using any of PyTorch's activation
        and pooling layers. The most flexible models are the rnn as they have the
        simplest setup.
        
        inputs:
            global
            - model_type (string): The model to build [linear, cnn, rnn]
            - inputs (int): The number of input neurons (linear/rnn) or channels (cnn)
            - outputs (int): The number of output neurons
            - neurons (list): A list of ints denoting the number of hidden neurons in each layer
            - activations (list): A list of strings denoting the names of the activation functions
            - linear_batch_normalization (bool): Should the linear layers be batch normalized?
            - linear_dropout (float): What percent of linear neurons be dropped?
            cnn
            - cnn_type (string): The cnn model to build [None, 1d, 2d]
            - channels (list): A list of ints denoting the number of output channels in each layer
            - image_width (int): The number of columns in the input array
            - image_height (int): The number of rows in the input array
            - kernels (list/tuple): A list of tuples or a tuple denoting the kernel size
            - strides (list/int): A list of ints or an int denoting the stride size
            - paddings (list/int): A list of ints or an int denoting the padding size
            - cnn_batch_normalization (bool): Should the cnn layers be batch normalized?
            - pooling (string): The name of the pooling function to use
            rnn
            - rnn_type (string): The rnn model to build [None, rnn, lstm, gru]
            - hidden_size (int): The number of hidden units each layer will have
            - num_layers (int): The number of rnn layers to create
            - bias (bool): Should the network use bias weights?
            - batch_first (bool): Is the batch the first dimension of the data?
            - dropout (float): The percent of neurons to dropout
            - bidirectional (bool): Is the network bidirectional?
            - proj_size (int): Only applies to LSTM projections
    """
    def create_model(self, 
                     model_type, inputs, outputs, neurons, activations, linear_batch_normalization=None, linear_dropout=None,
                     cnn_type=None, channels=None, image_width=None, image_height=None, kernels=None, strides=None, paddings=None, pooling=None, cnn_batch_normalization=None, cnn_dropout=None,
                     rnn_type=None, hidden_size=None, num_layers=None, bias=None, batch_first=None, rnn_dropout=None, bidirectional=None, proj_size=None):

        if model_type not in self.model_types:
            common.Print_Input_Warning('Model -> create model', self.model_types)
            return
        
        if not self.process_global_inputs(inputs, outputs, neurons, activations, linear_batch_normalization, linear_dropout):
            return
        
        elif model_type == 'linear':
            layers = len(neurons)
            self.report['inputs'] = inputs
            self.report['outputs'] = outputs
            self.report['neurons'] = neurons
            self.report['linear layers'] = layers
            self.report['linear batch normalization'] = True if linear_batch_normalization is None else linear_batch_normalization
            self.report['linear dropout'] = self.expand_to_list(0, layers) if linear_dropout is None else self.expand_to_list(linear_dropout, layers)
            self.expand_activations(activations, [layers])
            self.build_linear()
            
            summary(self.model, [(self.report['inputs'])])
                
        elif model_type == 'cnn':
            if self.process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, pooling, cnn_batch_normalization, cnn_dropout):
                layers = len(neurons)
                self.report['inputs'] = inputs
                self.report['outputs'] = outputs
                self.report['neurons'] = neurons
                self.report['linear layers'] = layers
                self.report['linear batch normalization'] = False if linear_batch_normalization is None else linear_batch_normalization
                self.report['linear dropout'] = self.expand_to_list(0,layers) if linear_dropout is None else self.expand_to_list(linear_dropout, layers)
                
                layers = len(channels)
                self.report['cnn type'] = cnn_type
                self.report['channels'] = channels
                self.report['cnn layers'] = layers
                self.report['image width'] = image_width
                self.report['image height'] = image_height
                self.report['kernels'] = self.expand_to_list(kernels, layers) if type(kernels) is tuple else kernels
                self.report['strides'] = self.expand_to_list(1, layers) if strides is None else self.expand_to_list(strides, layers)
                self.report['paddings'] = self.expand_to_list(0, layers) if paddings is None else self.expand_to_list(paddings, layers)
                self.report['pooling'] = pooling if pooling is None else self.set_pooling(pooling)
                print(self.report['pooling'])
                self.report['cnn batch normalization'] = False if cnn_batch_normalization is None else cnn_batch_normalization
                self.report['cnn dropout'] = self.expand_to_list(0, layers) if cnn_dropout is None else self.expand_to_list(cnn_dropout, layers)
                self.expand_activations(activations, [self.report['cnn layers'], self.report['linear layers']])
                self.build_cnn()
                
                summary(self.model, [(self.report['inputs'], self.report['image width'], self.report['image height'])])
            else:
                return
            
        
        elif model_type == 'rnn': 
            if self.process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, rnn_dropout, bidirectional, proj_size):
                layers = len(neurons)
                self.report['inputs'] = inputs
                self.report['outputs'] = outputs
                self.report['neurons'] = neurons
                self.report['linear layers'] = layers
                self.report['linear batch normalization'] = True if linear_batch_normalization is None else linear_batch_normalization
                self.report['linear dropout'] = self.expand_to_list(0, layers) if linear_dropout is None else self.expand_to_list(linear_dropout, layers)
                self.report['rnn type'] = rnn_type
                self.report['hidden size'] = hidden_size
                self.report['rnn layers'] = num_layers
                self.report['bias'] = True if bias is None else bias
                self.report['batch first'] = False if batch_first is None else batch_first
                self.report['rnn dropout'] = 0 if rnn_dropout is None else rnn_dropout
                self.report['bidirectional'] = False if bidirectional is None else bidirectional
                self.report['projection size'] = 0 if proj_size is None else proj_size
                self.expand_activations(activations, [layers])
                self.build_rnn()
                
                summary(self.model, (1, 1, self.report[inputs]))
            else:
                return
        
        
        
    
    """
        expand_activations(activations, length)
        A function to expand a list of 2 or 3 activation functions to equal the
        number of hidden layers specified in length. It is assumed that the order
        of activations is linear, output (2) or cnn, linear, output (3). The 
        resulting list of activation functions is stored in the report dictionary.
        
        inputs:
            - activations (list): A list of strings denoting the names of the activation functions
            - length (list): A list of ints specifing the number of layers using said activations
        outputs:
            - 
    """
    def expand_activations(self, activations, length):
        acts = []
        if len(activations) < sum(length):
            for i, value in enumerate(length):
                for j in range(value):
                    acts.append(self.set_activation(activations[i]))
            
            acts.append(self.set_activation(activations[-1]))
            
            self.report['activations'] = acts
        else:
            for activation in activations:
                acts.append(self.set_activation(activation))
            self.report['activations'] = acts
        
    
    """
        expand_to_list(value, length)
        A function to expand a value into a list of that value.
        
        inputs:
            - value (): The value to copy throughout the list
            - length (int): The length of the output list
        outputs:
            - output_list (list): The list of expanded values
    """
    def expand_to_list(self, value, length):
        output_list = []
        for i in range(length):
            output_list.append(value)
        return output_list
        
        
    """
        process_cnn_inputs(cnn_type, channels, image_width, image_height, kernels, strides, paddings, cnn_batch_normalization, pooling)
        A function to validate the inputs to build a convolution neural network. 
        If any of the inputs are invalid a boolean value of False is returned 
        and an appropriate error message is printed. See function create_model()
        for more information about what each variable is used for.
    """
    def process_cnn_inputs(self, cnn_type, channels, image_width, image_height, kernels, strides, paddings, pooling, cnn_batch_normalization, cnn_dropout):
        
        e = ''
        success = True
        
        if cnn_type not in self.cnn_types:
            e += f'cnn type not recognized\nRecieved {cnn_type}'
            common.Print_Input_Warning('Model -> process cnn inputs', self.cnn_types)
            success = False
        
        elif type(cnn_type) is not str or cnn_type is None:
            e += f'cnn type must be of type <str>\nBut Recieved {cnn_type}'
            success = False
            
        elif (type(channels) is not list and type(channels) is not int) or channels is None:
            e += f'channels must be of type <list> or <int>\nBut Recieved {channels}'
            success = False
        
        elif type(image_width) is not int or image_width is None:
            e += f'image width must be of type <int>\nBut Recieved {image_width}'
            success = False
        
        elif type(image_height) is not int or image_height is None:
            e += f'image height must be of type <int>\nBut Recieved {image_height}'
            success = False
        
        elif type(kernels) is not list and type(kernels) is not tuple and kernels is not None:
            e += f'kernels must be of type <list> or <tuple>\nBut Recieved {kernels}'
            success = False
        
        elif type(strides) is not list and type(strides) is not int and strides is not None:
            e += f'strides must be of type <list> or <int>\nBut Recieved {strides}'
            success = False
        
        elif type(paddings) is not list and type(paddings) is not int and paddings is not None:
            e += f'paddings must be of type <list> or <int>\nBut Recieved {paddings}'
            success = False
            
        elif type(pooling) is not str and pooling is not None:
            e += 'pooling must be of type <str>\nBut Recieved {pooling}'
            success = False
            
        elif type(cnn_batch_normalization) is not bool and cnn_batch_normalization is not None:
            e += f'cnn batch normalization must be of type <float>\nBut Recieved {cnn_batch_normalization}'
            success = False
            
        elif type(cnn_dropout) is not float and cnn_dropout is not None:
            e += f'cnn dropout must be of type <float>\nBut Recieved {cnn_dropout}'
            success = False
        
        if not success:
            common.Print_Error('Model -> process cnn inputs', e)
        
        return success
        
    
    """
        process_global_inputs(inputs, outputs, neurons, activations, batch_normalization, dropout)
        A function to validate the required inputs to build any network. If any
        of the inptus are invalid a boolean of False is returned and an 
        appropriate error message is printed. See function create_model() for
        more information about what each variable is used for.
    """
    def process_global_inputs(self, inputs, outputs, neurons, activations, batch_normalization, dropout):
        
        e = ''
        success = True
        
        if type(inputs) is not int or inputs < 1:
            e += f'inputs must be of type <int> and > 0\nBut recieved {inputs}'
            success = False
        
        elif type(outputs) is not int or outputs < 1:
            e += f'outputs must be of type <int> and > 0\nBut recieved {outputs}'
            success = False
            
        elif type(neurons) is not list:
            e += f'neurons must be of type <list>\nBut recieved {neurons}'
            success = False
        
        elif type(activations) is not list and type(activations) is not str:
            e += f'activations must be of type <list> or <str>\nBut recieved {activations}'
            success = False
        
        elif type(batch_normalization) is not bool and batch_normalization is not None:
            e += f'batch normalizations must be of type <float>\nBut recieved {batch_normalization}'
            success = False
        
        elif type(dropout) is not float and dropout is not None:
            e += f'dropout must be of type <list> or <float>\nBut recieved {dropout}'
            success = False
            
        if not success:
            common.Print_Error('Model -> process global inputs', e)
        
        return success
        
        
    """
        process_rnn_inputs(rnn_type, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        A function to validate the inputs to build an rnn neural network. If any
        of the inputs are invalid a boolean value of False is returned and an 
        appropriate error message is printed. See function create_model() for 
        more information about what each variable is used for.
    """
    def process_rnn_inputs(self, rnn_type, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size):
        
        e = ''
        success = True
        
        if rnn_type not in self.rnn_types:
            e += f'rnn type not recognized\nRecieved {rnn_type}'
            common.Print_Input_Warning('Model -> process rnn inputs', self.rnn_types)
            success = False
            
        elif type(rnn_type) is not str or rnn_type is None:
            e += f'rnn type must be of type <str>\nBut Recieved {rnn_type}'
            success = False
            
        elif type(hidden_size) is not int or hidden_size is None:
            e += f'hidden size must be of type <int>\nBut Recieved {hidden_size}'
            success = False
            
        elif type(num_layers) is not int or num_layers is None:
            e += f'num layers must be of type <int>\nBut Recieved {num_layers}'
            success = False
            
        elif type(bias) is not bool and bias is not None:
            e += f'bias must be of type <bool>\nBut Recieved {bias}'
            success = False
            
        elif type(batch_first) is not bool and batch_first is not None:
            e += f'batch first must be of type <bool>\nBut Recieved {batch_first}'
            success = False
            
        elif type(dropout) is not float and dropout is not None:
            e += f'dropout must be of type <float>\nBut Recieved {dropout}'
            success = False
            
        elif type(bidirectional) is not bool and bidirectional is not None:
            e += f'bidirectional must be of type <bool>\nBut Recieved {bidirectional}'
            success = False
            
        elif type(proj_size) is not int and proj_size is not None:
            e += f'proj size must be of type <int>\nBut Recieved {proj_size}'
            success = False
        
        if not success:
            common.Print_Error('Model -> process inputs rnn', e)
        
        return success
            
        
    """
        set_activation(name)
        A function to set the activation type in the report dictionary. The 
        input name must be of type <str> and the defualt value is ReLU
        
        inputs:
            - name (string): The name of the function to use
        outputs:
            - activation (reference): A reference to the activation function
    """
    def set_activation(self, name):
        name = name.lower()
        activations = {'elu':nn.ELU,
                       'hardshrink':nn.Hardshrink,
                       'hardsigmoid':nn.Hardsigmoid,
                       'hardtanh':nn.Hardtanh,
                       'hardswish':nn.Hardswish,
                       'leakyrelu':nn.LeakyReLU,
                       'logsigmoid':nn.LogSigmoid,
                       'multiheadattention':nn.MultiheadAttention,
                       'prelu':nn.PReLU,
                       'relu':nn.ReLU,
                       'relu6':nn.ReLU6,
                       'rrelu':nn.RReLU,
                       'selu':nn.SELU,
                       'celu':nn.CELU,
                       'gelu':nn.GELU,
                       'sigmoid':nn.Sigmoid,
                       'silu':nn.SiLU,
                       'mish':nn.Mish,
                       'softplus':nn.Softplus,
                       'softshrink':nn.Softshrink,
                       'softsign':nn.Softsign,
                       'tanh':nn.Tanh,
                       'tanhshrink':nn.Tanhshrink,
                       'threshold':nn.Threshold,
                       'glu':nn.GLU,
                       'softmin':nn.Softmin,
                       'softmax':nn.Softmax,
                       'softmax2d':nn.Softmax2d,
                       'logsoftmax':nn.LogSoftmax,
                       'adaptivelogsoftmaxwithloss':nn.AdaptiveLogSoftmaxWithLoss}
        
        activation = nn.ReLU
        for key in activations.keys():
            if key == name:
                activation = activations[key]
                break
                
        return activation
    
    
    """
        set_pooling(name)
        A function to set the pooling function in the report dictionary. A 
        pooling layer is not required so name can be None, in which case the 
        majority of the function is skipped.
        
        inputs:
            - name (string): The name of the pooling function to use
        outputs:
            -
    """
    def set_pooling(self, name): 
        if name is None:
            self.report['pooling'] = None
        else:
            name = name.lower()
            poolers = {'maxpool1d':nn.MaxPool1d,
                       'maxpool2d':nn.MaxPool2d,
                       'maxpool3d':nn.MaxPool3d,
                       'avgpool1d':nn.AvgPool1d,
                       'avgpool2d':nn.AvgPool2d,
                       'avgpool3d':nn.AvgPool3d,
                       'fractionalmaxpool2d':nn.FractionalMaxPool2d,
                       'fractionalmaxpool3d':nn.FractionalMaxPool3d,
                       'lppool1d':nn.LPPool1d,
                       'lppool2d':nn.LPPool2d,
                       'adaptivemaxpool1d':nn.AdaptiveMaxPool1d,
                       'adaptivemaxpool2d':nn.AdaptiveMaxPool2d,
                       'adaptivemaxpool3d':nn.AdaptiveMaxPool3d,
                       'adaptiveavgpool1d':nn.AdaptiveAvgPool1d,
                       'adaptiveavgpool2d':nn.AdaptiveAvgPool2d,
                       'adaptiveavgpool3d':nn.AdaptiveAvgPool3d}
        
            pooler = None
            for key in poolers.keys():
                if key == name:
                    pooler = poolers[key]
                    break

            return pooler
        
