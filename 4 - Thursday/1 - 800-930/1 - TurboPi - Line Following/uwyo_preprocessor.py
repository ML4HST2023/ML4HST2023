# -*- coding: utf-8 -*-
"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/01/2023

Purpose:
Used to provide common functionality such as progress bars, error reporting, and folder/file validation.

Functions:
 - Build_Confusion_Matrix_Output(confusion_matrix)
 - Build_Outputs(report):
 - Condense_History(history)
 - Create_Separator(message, character='=')
 - Generate_Confusion_Matrix(data, predictions)
 - Generate_Results_Graphs(file_name, data, predictions, time_steps)
 - Print(message)
 - Print_Error(function, message)
 - Print_Input_Warning(function, options)
 - Print_Message(message, variables=None)
 - Print_Status(name, value, total, history=None)
 - Prompt_For_Input(prompt='Prompt Unchanged', options=[])
 - Save_CSV(file_name, x, y, header)
 - Save_History(file_name, history)
 - Save_Model(file_name, model, base_name=None)
 - Save_Report(file_name, report)
 - Save_Results(file_name, trainer_report, model_report, confusion_matrix)
 - Update_Confusion_Matrix(confusion_matrix, prediction, target, time_steps)
 - Validate_Dir(directory)
 - Validate_File(file)

Included with: 
 - uwyo_common.py
 - uwyo_crop.py 
 - uwyo_MNIST.py
 - uwyo_models.py
 - uwyo_preprocessor.py (current file)
 - uwyo_train_detector.py
 - uwyo_trainer.py
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

# Outside Dependencies
import os
import glob
import cv2
import torch
from random import randint        

# In House Dependencies
import uwyo_common as common


__name__ = 'uwyo_preprocessor'
   

"""
    preprocessor(name='default')
    A class to provide image processing tasks such as finding the mean and std 
    of a dataset, cropping, and distribution into training, validation, and 
    testing datasets.
    
    inputs:
        - name (string): An identifier to differentiate between preprocessors
    outputs:
        -    
"""
class preprocessor():
    def __init__(self, name='default'):        
        self.name = name

    """
        Calculate_Mean_Std(path)
        A function to calculate the average mean and standard deviation of an 
        image dataset. This information can be used to normalize an image dataset.
        
        inputs:
            - path (string): The relative location of the data to process
        outputs:
            - mean (float): The average mean of the dataset
            - std (float): The average std of the dataset
    """
    def Calculate_Mean_Std(self, path):
        try:
            file_list = glob.glob(os.path.join(path, '*.jpg'))    
            file_count = len(file_list)
            mean = torch.tensor([0,0,0])
            std = torch.tensor([0,0,0])
        
            for file in file_list:
                image = torch.tensor(cv2.imread(file), dtype=torch.float32)
            
                image_mean = image.mean(dim=0)
                image_mean = image_mean.mean(dim=0)
                mean = torch.add(image_mean, mean)
            
                image_std = image.std(dim=0)
                image_std = image_std.std(dim=0)
                std = torch.add(image_std, std)
            
            mean = torch.div(mean,file_count) / 255
            std = torch.div(std,file_count) / 255
        
            return mean, std
        
        except Exception as e:
            common.Print_Error('Preprocessor -> Calculate Mean STD', e)    
    
    
    """
        Copy_Crop(old_locatiion, new_location, crop_location)
        A function to copy an image, crop the image, and then save the cropped
        images to a new location.
        
        inputs:
            - old_location (string): The location of the data now
            - new_location (string): The location of the cropped data
            - crop_location (string): An identifier used to differentiate between cropped images and raw images
        outputs:
            -
    """
    def Copy_Crop(self, old_location, new_location, divisions=4, level=1):
        try:
            file_list = glob.glob(os.path.join(old_location, '*.jpg'))    
            image_count = len(file_list)
            crop_count = image_count * divisions * level
        
            print(f'[INFO] Processing {image_count} images from {old_location}')
            print(f'[INFO] Creating {crop_count} images at {new_location}')
            file_check = glob.glob(os.path.join(new_location, '*jpg'))
            common.Validate_Dir(new_location)
            
            if len(file_check) < 1:
                crop_image = []
                    
                for index in range(image_count):
                    image = cv2.imread(file_list[index])
                    crop_image += self.Crop_Image(image, divisions, level)
                    common.Print_Status('Crop', index, image_count)
                
                for index, image in enumerate(crop_image):
                    cv2.imwrite(f'{new_location}{index}.jpg', image)
                    common.Print_Status('Copy', index, crop_count)
                
            else:
                print(f'[WARNING] {len(file_check)} Image Files Are Already Present At The Specified Location {new_location}, Stopping Copy Process...')
                
        except Exception as e:
            common.Print_Error('preprocessor -> Copy Crop', e)
    
    
    """
        Crop_Image(location, image, divisions=4, level=1)
        A function to crop an image into sub-images. The sub images start
        at the bottom of the image (level = 1) and progress up the image. If 
        level <= 3 the divisions are of size 56 and the image will not be 
        completely represented in the sub images.
        
        inputs:
            - image (numpy array): The image as an array
            - divisions (int): The number of times to divide the image width
            - level (int): The number of times to divide the image height
    """
    def Crop_Image(self, image, divisions=4, level=1):
        try:        
            crops = []
            width = image.shape[1]
            height = image.shape[0]
            lev_size = int(image.shape[0] / level) if level > 4 else int(image.shape[0] / 4)
            div_size = int(image.shape[1] / divisions)
            
            for i in range(level):
                ni = i + 1
                for j in range(divisions):
                    nj = j + 1
                    crop = image[height-(ni*lev_size):height-(i*lev_size),j*div_size:nj*div_size]
                    crops.append(crop)
            return crops
        except Exception as e:
            common.Print_Error('preprocessor -> Crop Image', e)
    
    
    """
        Distribute_Data(original_location, training_location, testing_location, split)
        A function to split the data into training and testing folders based
        on the split variable.
        
        inputs:
            - original_location (string): The location of the data now
            - training_location (string): The location of the training data
            - testing_location (string): The location of the testing data
            - split (float): The percent of data in the training set 
    """
    def Distribute_Data(self, original_location, training_location, testing_location, split, validation_location=None):
        try:
            if split < 1 and split > 0:
                split *= 100
            
            training_location += 'positive/' if 'positive' in original_location else 'negative/'
            testing_location += 'positive/' if 'positive' in original_location else 'negative/'
            
            common.Validate_Dir(training_location)
            common.Validate_Dir(testing_location)
            
            if validation_location is not None:
                validation_location += 'positive/' if 'positive' in original_location else 'negative/'
                common.Validate_Dir(validation_location)
            
            file_list = glob.glob(os.path.join(original_location, '*.jpg'))
            file_count = len(file_list)
            
            common.Print(f'[INFO] Found {file_count} files at {original_location}')
        
            train_file_check = glob.glob(os.path.join(training_location, '*.jpg'))
            test_file_check = glob.glob(os.path.join(testing_location, '*.jpg'))
            
            valid_file_check = [] if validation_location is None else glob.glob(os.path.join(validation_location, '*.jpg'))
            
            file_check = len(train_file_check) + len(test_file_check) + len(valid_file_check)
            
            if file_check < 1:
                train_files = []
                valid_files = []
                test_files = []
                temp_test_files = []
                for i, file in enumerate(file_list):
                    rand = randint(0, 100)
                    if rand < split:
                        train_files.append(file)
                    else:
                        temp_test_files.append(file)
                    common.Print_Status('Distribute Data', i, file_count)
                
                if valid_file_check is not None:
                    for file in temp_test_files:
                        rand = randint(0, 100)
                        if rand < 50:
                            valid_files.append(file)
                        else:
                            test_files.append(file)
                else:
                    test_files = temp_test_files
                
                for i, file in enumerate(train_files):
                    cv2.imwrite(training_location + str(i) + '.jpg', cv2.imread(file))
                    
                for i, file in enumerate(valid_files):
                    cv2.imwrite(validation_location + str(i) + '.jpg', cv2.imread(file))
                
                for i, file in enumerate(test_files):
                    cv2.imwrite(testing_location + str(i) + '.jpg', cv2.imread(file))
                
            else:
                if len(train_file_check) > 0:
                    location = training_location
                elif len(test_file_check) > 0:
                    location = testing_location
                elif len(valid_file_check) > 0:
                    location = validation_location
                    
                print(f'[WARNING] {file_check} Image Files Are Already Present At The Specified Location {location} , Stopping Distribution Process...')
        
        except Exception as e:
            common.Print_Error('preprocessor -> Distribute Data', e)
    
    