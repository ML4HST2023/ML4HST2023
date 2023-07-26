# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:50:46 2023

@author: jblaney1
"""


import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


import uwyo_common as common


__name__ = 'uwyo_dataloader'


"""
    load_data(path, batch_siz, image_width, image_height, transform_train, transform_test, valid, display)
    A function to automate loading image datasets to be used for training and testing
    pytorch image models. If the image folder is not divided into training and testing 
    or training and validation and testing, then it is assumes the images are sorted by 
    class and will be divided into an 80 - 20 or 80 - 10 - 10 split depending on valid
    boolean value.
    
    inputs:
     - path (string): Path to the base folder where the images have been sorted.
     - batch_size (int): The number of examples per batch
     - image_width (int): The transformed image width
     - image_height (int): The transformed image height
     - transform_train (pytorch transform): A transformation to use when loading training images
     - transform_test (pytorch transform): A tranformation to use when loading testing images
     - valid (boolean): Is there a validation dataset?
     - display (boolean): Should examples be shown?
    outputs:
     - train_dl (pytorch dataloader): The dataloader for the training dataset
     - valid_dl (pytorch dataloader): The dataloader for the validation dataset
     - tests_dl (pytorch dataloader): The dataloader for the testing dataset
     - labels (list): The strings representing the text labels
"""
def load_images(path, batch_size=64, image_width=64, image_height=64, transform_train=None, transform_test=None, valid=True, display=False):
    # Transform to be used on our training/validation dataset if none were specified
    if transform_train is None:
        transform_train = transforms.Compose([transforms.Resize([image_width, image_height]),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(), ])
    
    # Transform to be used on our testing dataset if none were specified
    if transform_test is None:
        transform_test = transforms.Compose([transforms.Resize([image_width, image_height]),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(), ])
    
    dictionary = {'label name':[],'files':[],'label int':[]}
    
    try:
        # Create datasets from the specified presplit path
        train_dataset = datasets.ImageFolder(path + '/training/', transform=transform_train)
        valid_dataset = datasets.ImageFolder(path + '/validation/', transform=transform_train) if valid else None
        tests_dataset = datasets.ImageFolder(path + '/testing/', transform=transform_test)   
        images = train_dataset.imgs
    except:
        # Create a single dataset from the specified path to split later
        dataset = datasets.ImageFolder(path, transform=transform_train)
        generator = torch.Generator().manual_seed(42)
        if valid:
            train_dataset, valid_dataset, tests_dataset = random_split(dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=generator)
        else:
            train_dataset, tests_dataset = random_split(dataset=dataset, lengths=[0.8, 0.2], generator=generator)
        images = dataset.imgs
    
    # Strip the labels from the dataloader object
    for image in images:
        if image[1] not in dictionary['label int']:
            dictionary['files'].append(image[0])
            dictionary['label int'].append(image[1])
            label_split = image[0].split('\\')
            label_name = label_split[1] if len(label_split) > 2 else label_split[0].split('/')[-1]
            dictionary['label name'].append(label_name)
        
    # Printing out the lengths of our datasets:
    message = f'Training dataset length: {len(train_dataset)}\n'
    message += f'Validation dataset length: {len(valid_dataset)}\n' if valid else ''
    message += f'Testing dataset length: {len(tests_dataset)}'
    common.Print(message)
    
    # Training Dataloader
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)

    # Validation Dataloader
    valid_dl = DataLoader(valid_dataset, batch_size, shuffle=False) if valid else None

    # Testing Dataloader
    tests_dl = DataLoader(tests_dataset, len(tests_dataset), shuffle=False)

    # Print how many batches will be in our training/validation dataloaders (dependent on batch_size)
    message = f'Number of training batches: {len(train_dl)}\n'
    message += f'Number of validation batches: {len(valid_dl)}\n' if valid else ''
    message += f'Number of testing batches: {len(tests_dl)}'
    common.Print(message)
    
    labels = dictionary['label name']
    # Display examples from each class
    if display:
        cols = 5
        rows = int(len(labels) / cols) + 1
        fig = plt.figure(figsize=(2*cols,2*rows))
        for index, file in enumerate(dictionary['files']):
            image = Image.open(file)
            sub_plot = fig.add_subplot(rows, cols, index+1)
            sub_plot.set_title(f'{index} {labels[index]}')
            sub_plot.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            plt.imshow(image)
        
    # Display the mapping from int to string for each class
    else:
        message = 'Labels\n'
        for index, label in enumerate(labels):
            message += f'{index} -> {label}\n'
        common.Print(message)
    
    # Return a validation dataloader
    if valid:
        return train_dl, valid_dl, tests_dl, labels
    
    # Do not return a validation dataloader
    else:
        return train_dl, tests_dl, labels