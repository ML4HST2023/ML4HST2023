# -*- coding: utf-8 -*-
"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/01/2023

Purpose:


Functions:
 - __init__(name, model, device='cuda:0')
 - attribute_get(name)
 - attribute_set(name)
 - plot_history(history, multiplot=False)
 - set_loss(name)
 - set_optimizer(name)
 - test(test)
 - train(num_epochs, train, valid=None)

Included with: 
 - common.py
 - model.py
 - trainer.py (current file)
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

# Outside Dependencies
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# In House Dependencies
import uwyo_common as common

__name__ = 'uwyo_trainer'


"""
    plot_history(history)
    A function to automate plotting all of the metrics tracked in the 
    history dictionary during training. If a validation dataset was used
    the validation metrics are plotted on the save graph as the training
    metrics.
    
    inputs:
     - history (dictionary): The dictionary of metrics from training
    outputs:
     -
"""
def plot_history(history):
    keys = list(history.keys())
    epochs = len(history[keys[0]])
    x = range(epochs)
    val = 'val_loss' in keys
    num_keys = int(len(keys)/2) if val else len(keys)
    fig, ax = plt.subplots(1, num_keys, figsize=(12,4))
    
    for i in range(num_keys):
        y1 = history[keys[i]]
        ax[i].plot(x, y1, label='Train')
        if val:
            y2 = history[keys[i + int(len(keys)/2)]]
            ax[i].plot(x, y2, label='Valid')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(keys[i])
        ax[i].grid(True)
    
    plt.show()
        

"""
    set_loss_fn(name)
    A function to wrap the available pytorch loss functions so that
    the training function can reference the loss function using a 
    string. For information on how each loss function works please
    refer to pytorch's online documentation.
    
    inputs:
     - name (string): The name of the loss function to use
    outputs:
     - loss (reference): A reference to the pytorch loss function
"""
def set_loss_fn(name):
    try:
        name = name.lower()
        losses = {'l1':nn.L1Loss,
                 'mse':nn.MSELoss,
                 'crossentropy':nn.CrossEntropyLoss,
                 'ctc':nn.CTCLoss,
                 'nll':nn.NLLLoss,
                 'poissonnll':nn.PoissonNLLLoss,
                 'gaussiannll':nn.GaussianNLLLoss,
                 'kldiv':nn.KLDivLoss,
                 'bce':nn.BCELoss,
                 'bcewithlogits':nn.BCEWithLogitsLoss,
                 'marginranking':nn.MarginRankingLoss,
                 'hingeembedding':nn.HingeEmbeddingLoss,
                 'multilabelmargin':nn.MultiLabelMarginLoss,
                 'huber':nn.HuberLoss,
                 'smoothl1':nn.SmoothL1Loss,
                 'softmargin':nn.SoftMarginLoss,
                 'multilabelsoftmargin':nn.MultiLabelSoftMarginLoss,
                 'cosineembedding':nn.CosineEmbeddingLoss,
                 'multimargin':nn.MultiMarginLoss,
                 'tripletmargin':nn.TripletMarginLoss,
                 'tribletmarginwithdistance':nn.TripletMarginWithDistanceLoss}

        loss = nn.MSELoss
        for key in losses.keys():
            if key == name:
                loss = losses[key]
                break

        return loss

    except Exception as e:
        common.Print_Error('GAN -> set loss', e)
    

"""
    set_optimizer(name)
    A function to wrap the available pytorch optimizer algorithms
    so that the training function can reference the optimizer using 
    a string. For information on how each optimizer works please 
    refer to pytorch's online documentation.
    
    inputs:
     - name (string): The name of the optimization algorithm to use
    outputs:
     - optimizer (reference): A reference to the pytorch optimizer
"""
def set_optimizer(name):
    try:
        name = name.lower()
        optimizers = {'adadelta':optim.Adadelta,
                      'adagrad':optim.Adagrad,
                      'adam':optim.Adam,
                      'adamw':optim.AdamW,
                      'sparseadam':optim.SparseAdam,
                      'adamax':optim.Adamax,
                      'asgd':optim.ASGD,
                      'lbfgs':optim.LBFGS,
                      'nadam':optim.NAdam,
                      'radam':optim.RAdam,
                      'rmsprop':optim.RMSprop,
                      'rprop':optim.Rprop,
                      'sgd':optim.SGD}

        optimizer = optim.SGD
        for key in optimizers.keys():
            if key == name:
                optimizer = optimizers[key]
                break
                
        return optimizer
    
    except Exception as e:
        common.Print_Error('GAN -> set optimizer', e)
        
    
"""
    test(mode, tests_dl, labels, thresh, verbose, device)
    A function to automate the testing procedure for pytorch models. If verbose
    is greater than 1 it is assumed that there is only one batch in the tests_dl
    dataloader.
    
    inputs:
     - model (pytorch model): The model to test
     - tests_dl (pytorch dataloader): The data to use for testing
     - labels (list): A list of strings specifying the text labels
     - thresh (float): The threshold for classification, only used for binary classification
     - verbose (int): The level of output to print
     - device (string): The device to perform model processing on
    outputs:
     - accuracy (float): The models accuracy over the test dataset
"""
def test(model, tests_dl, labels=None, thresh=0.5, verbose=0, device='cuda:0'):
    # Transfer model to the device
    device = torch.device(device)
    dev_model = model.to(device)
    
    # Start the data count and accuracy at zero
    data_count = 0
    accuracy = 0.0

    # Set the trained model for evaluation and disable gradient updates
    dev_model.eval()
    with torch.no_grad():
        for x_batch, y_batch in tests_dl: 
            # Transfer batched data to the same device as the model
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Predict a batch of outputs using the model
            pred = dev_model(x_batch).squeeze()
            
            # Threshold or one-hot the prediction according to the prediction shape
            pred = torch.argmax(pred, dim=1) if len(pred.shape) > 1 else (pred >= thresh).float()
            is_correct = (pred == y_batch).float() # Tabulate the number of correct predictions
            accuracy += is_correct.sum().cpu() # Add the number of correct outputs to the accuracy
            data_count += len(y_batch) # Add the number of inputs in this batch to the processed data count
            
            # If more information is desired output class information
            if verbose > 0:
                # Transfer the data to the cpu
                y_batch = y_batch.cpu()
                pred = pred.cpu()
                
                # If text labels have been provided, use them
                if labels is not None:
                    print(classification_report(y_batch, pred, target_names=labels))
                    conf_mat = confusion_matrix(y_batch, pred, labels=y_batch.unique())
                    conf_disp = ConfusionMatrixDisplay(conf_mat, labels)
                else:
                    print(classification_report(y_batch, pred))
                    conf_mat = confusion_matrix(y_batch, pred, labels=y_batch.unique())
                    conf_disp = ConfusionMatrixDisplay(conf_mat, y_batch.unique())
                    
                conf_disp.plot(xticks_rotation='vertical')
    # Normalize the accuracy to [0 - 1]
    accuracy /= data_count
    print(f'Test Dataset Accuracy: {accuracy}')
    
    return accuracy

    
"""
    train(model, num_epochs, train_dl, valid_dl=None, optimizer='adam', loss_fn='mse', lr=0.001, device='cuda:0')
        A function to automate pytorch model training.
        
    inputs:
     - model (pytorch model): The pytorch model to train
     - num_epochs (int): The number of epochs to train the model for
     - train_dl (dataloader): Pytorch dataloader object for training dataset
     - valid_dl (dataloader): Pytorch dataloader object for validation dataset
     - optimizer (string): The name of the optimizer function to use. See set_optimizer() for available functions
     - loss_fn (string): The name of the loss function to use. See set_loss_fn() for available functions
     - lr (float): The learning rate for the optimizer to use
     - thresh (float): The threshold to consider an answer as class 1 (lower) or class 2 (higher)
     - device (float): The name of the device to train on
    outputs:
     - history (dictionary): A dictionary of the training metrics
     - dev_model (pytorch model): The trained model on the cpu
"""
def train(model, num_epochs, train_dl, valid_dl=None, optimizer='adam', loss_fn='mse', lr=0.001, thresh=0.5, device='cuda:0'):
    
    train_count = len(train_dl.dataset)
    valid_count = 0 if valid_dl is None else len(valid_dl.dataset)
    
    # Transfer the model to the device
    device = torch.device(device)
    dev_model = model.to(device)

    # A dictionary which will house metrics for both the training and validation datasets
    history = {}
    metrics = ['loss', 'acc'] if valid_dl is None else ['loss', 'acc', 'val_loss', 'val_acc']
    for name in metrics:
        history[name] = [0] * num_epochs
    
    # Get the function reference for the pytorch loss function
    loss_fn = set_loss_fn(loss_fn)()
    
    # Get the function reference for the pytorch optimizer
    optimizer = set_optimizer(optimizer)(dev_model.parameters(), lr=lr)
    
    # Initiate training for our planned number of epochs
    for epoch in range(num_epochs):
        dev_model.train() # Set the model to training mode (compute gradients)

        # Iterate through our batches (housed within our training DataLoader object)
        for x_batch, y_batch in train_dl:
            # Transfer batched data to the same device as the model
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Predict a batch of outputs using the model
            pred = dev_model(x_batch).squeeze()

            # Compute the loss
            loss = loss_fn(pred, y_batch.float())

            # Backpropagate our loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute any training metrics
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float() if len(pred.shape) > 1 else ((pred >= thresh).float() == y_batch).float()
            history['loss'][epoch] += loss.item()*y_batch.size(0)
            history['acc'][epoch] += is_correct.sum().cpu()
        
        # Compute the mean of the training metrics
        history['loss'][epoch] /= train_count
        history['acc'][epoch] /= train_count

        # Perform validation
        if valid_dl is not None:
            # Set the model for evaluation and disable gradient updates
            dev_model.eval() 
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    # Transfer batched data to the same device as the model
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Predict a batch of outputs using the model
                    pred = dev_model(x_batch).squeeze()

                    # Compute the loss
                    loss = loss_fn(pred, y_batch.long())

                    # Compute any validation metrics
                    history['val_loss'][epoch] += loss.item()*y_batch.size(0)
                    is_correct = (torch.argmax(pred, dim=1) == y_batch).float() if len(pred.shape) > 1 else ((pred >= thresh).float() == y_batch).float()
                    history['val_acc'][epoch] += is_correct.sum().cpu()
            
            # Compute the mean of the validation metrics
            history['val_loss'][epoch] /= valid_count
            history['val_acc'][epoch] /= valid_count
        
        # Update the progress bar for training
        common.Print_Status('Training', epoch, num_epochs, history)
        
    print()
    return history, dev_model.to('cpu') 
