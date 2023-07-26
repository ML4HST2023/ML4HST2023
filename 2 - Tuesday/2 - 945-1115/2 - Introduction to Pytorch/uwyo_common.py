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
 - uwyo_common.py (current file)
 - uwyo_crop.py
 - uwyo_MNIST.py
 - uwyo_models.py
 - uwyo_preprocessor.py
 - uwyo_train_detector.py
 - uwyo_trainer.py
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

# Outside Dependencies
import os 
import pickle
import numpy as np
import pandas as pd

__name__='uwyo_common'


"""
    Build_Confusion_Matrix_Output(confusion_matrix)
    A function to automate the transfer of confusion matrices to text file ready strings.
    
    inputs:
        - confusion_matrix (numpy array): A 2d array representing the confusion matrix
    outputs:
        - output (string): A text file ready output documenting the confusion matrix
"""
def Build_Confusion_Matrix_Output(confusion_matrix):
    output = "\n"
    for row in confusion_matrix:
        output += " "
        for entry in row:
            output += f'{entry:.4f}' + "\t"
        output += "\n" 

    return output
        

"""

"""
def Build_History_Output(history, epoch):
    message = ''
    for key in history.keys():
        message += f'{key} : {history[key][epoch]:.4f} | '
    return message


"""
    Build_Outputs(report)
    A function to automate the tranformation from a report dictionary to an text file ready string.
    
    inputs:
        - report (dictionary): An ANN_Model or Trainer report dictionary 
    outputs:
        - output (string): A text file ready output documenting the object report information 
"""
def Build_Outputs(report):
    output = "\n"
    keys = report.keys()
    
    for key in keys:
        try:
            if type(report[key]) is int:
                output += str(key) + ": " + str(report[key]) + " | "
            elif type(report[key]) is float:
                output += str(key) + ": " + f'{report[key]:.4f}' + " | "
            elif type(report[key]) is list:
                if len(report[key]) > 0:
                    output += str(key) + ": " + str(report[key]) + " | "
            else:
                output += str(key) + ":" + str(report[key]) + " | "
        except Exception as e:
            Print_Error('Build Outputs', e)

    return output
        
   
"""
    Condense_History(history)
    A function to automate history averaging for default training.
    
    inputs:
        - history (dictionary): A dictionary of from TF history
    outputs:
        - output (dictionary): An averaged dictionary of history values 
"""
def Condense_History(history):
    epochs = len(history)
    output = {}
    keys = history.keys()
    for key in keys:
        output[key] = sum(history[key])/epochs
    return output


"""
    Create_Separator(message, character='=')
    A function to automate the creation of messaging separators which
    create nice printed outputs to deliniate the start and end of messages.
    Newline characters are used to denote breakpoints and to find the longest
    line in the message. Which will be the length of the separator.
    
    inputs:
        - message (string): The message to display
        - character (string): The character used to build the separator
    outputs:
        - separator (string): The separator of length equal to the longest line
"""     
def Create_Separator(message, character='='):
    separator_length = 0
    split_message = str(message).split("\n")
    for split in split_message:
        length = len(split)
        if length > separator_length:
            separator_length = length
    
    separator = ""
    for index in range(separator_length):
        separator += character
    
    return separator
    
    
"""
    Generate_Confusion_Matrix(data, predictions)
    A function to automate the generation of confusion matrices.
    
    inputs:
        - test_data (pandas dataframe): A dataframe of labels to use for targets 
        - predictions (numpy array): Model predictions from testing
    outputs:
        - predictions (list): A list of 2d numpy arrays for confusion matrices
        - (boolean): Was the operation successful?
"""
def Generate_Confusion_Matrix(data, predictions):
    predictions = []
    labels = data["labels"]
    try:
        for i, prediction in enumerate(predictions):
            predictions.append(Update_Confusion_Matrix(prediction=prediction, target=labels[i]))
        return predictions, True
    
    except Exception as e:
        Print_Error('Generate Confusion Matrix', e)
        return predictions, False


"""
    Generate_Results_Graphs(file_name, data, predictions, time_steps,)
    A function to automate the generation of prediction csv output files.
   
    inputs:
        - file_name (string): The file name to store the resultant CSVs under
        - data (pandas dataframe): The data to be stored in the resultant CSVs
        - predictions (pandas dataframe): The models predicted values offset by time_steps
        - time_steps (int): The time to offset the predictions from the data
    outputs:
        - (boolean): Was the operation successful?
"""
def Generate_Results_Graphs(file_name, data, predictions, time_steps):
    labels = data["labels"]
    features = data["raw"]
    time_steps -= 1
    
    try:
        for i, prediction in enumerate(predictions):
            label = labels[i][time_steps:]
            feature = features[i][time_steps:]
            df = pd.DataFrame({'feature':feature,'prediction':prediction,'label':label})
            df.to_csv(file_name + str(i) + '.csv', index=False)

        return True
    
    except Exception as e:
        Print_Error('Generate Results Graphs', e)
        return False            
    
    
"""
    Print(message)
    A function to provide structured print statements which are delineated by separators
    
    inputs:
        - message (string): The message to print
    ouputs:
        - 
"""
def Print(message):
    separator = Create_Separator(message)
    print("")
    print(separator)
    print(message)
    print(separator)
    print("")

    
"""
    Print_Error(function, message)
    A function which prints an error message stating which function generated the message
    followed by the error.
 
    inputs: 
        - function (string): The name of the calling function
        - message (string): The automatic or custom error message
    outputs:
        - 
"""
def Print_Error(function, message):
    message = f'[ERROR] The following error occured in function: {function} \n {message}'
    Print(message)    
    

"""

"""
def Print_History(name, history):
    message = Build_History_Output(history)
    print(f'{name} History | {message}')
    
    
"""
    Print_Input_Warning(function, options)
    A function to prompt the user when an error is thrown due to issues
    with one of their inputs.
    
    inputs:
        - function (string): The function where the error occured
        - options (list): The expected inputs
    outputs:
        -
"""
def Print_Input_Warning(function, options):
    message = '[WARNING] An invalid input was recieved in function : {function} \n'
    message += '         The valid inputs are enumerated below.'
    for index, option in enumerate(options):
        message += f' ({index}) {option} \n'
    Print(message)


"""
    Print_Message(message, variables=None):
    A function to print messages with the option of including a dictionary
    of variables to print.
    
    inputs:
        - message (string): The message to print
        - variables (dictionary): The variables to print below the message
    outputs:
        -
"""
def Print_Message(message, variables=None):
    if variables is not None:
        message += '\n'
        for key in variables.keys:
            message += f'{key} : {variables[key]} \n'
    Print(message)
        
    
"""
    Print_Status(name, index, total, history=None)
    A function which displays a status bar of the form below. If history is provided
    the average values will be computed and displayed.
    
    {name} index / total [==         ] 25% | history
    
    inputs:
        - name (string): The name of the calling process
        - index (int): The number of items processed
        - total (int): The number of items to process
        - history (dictionary): The history output from keras.model.fit()
    outputs:
        - 
"""
def Print_Status(name, value, total, history=None):
    value += 1
    percent_int = int(value/total*100)
    percent_string = ''

    for index in range(int(percent_int/5)):
        percent_string += '='

    if value <= total:
        if history is not None:
            message = Build_History_Output(history, value-1)
            print('\r', f'{value:>5} / {total:>5} | [{percent_string:<20}] {percent_int:>4}% \t | {message}', end="\r")
        else:
            print('\r', f'{value:>5} / {total:>5} | [{percent_string:<20}] {percent_int:>4}% ', end="\r")
    else:
        if history is not None:
            Print_History(name, history)
        else:
            print(f'{name} Complete [{percent_string}] {percent_int:>4}%')
        
        
"""
    Prompt_For_Input(prompt, options)
    Displays a prompt for input with the specified prompt followed by an enumerated
    list of options.
    
    inputs:
        - prompt (string): A message to be displayed before the list of options
        - options (list): A list of options which the user can select from
    outputs:
        -
"""
def Prompt_For_Input(prompt="Prompt Unchanged", options=[]):
    message = prompt + "\n"
    
    for index, option in enumerate(options):
        message += f' ({index}) - {option} \n'
    
    Print_Message(message)
    
    
"""
    Save_CSV(file_name, x, y, header)
    A function to save a feature (x) and a label (y) under the heading
    stored in header and the heading "Label" respectively.
    
    inputs:
        - file_name (string): The name of the CSV file to save the columns in
        - x (array): The feature column
        - y (array): The label column
        - header (string): The header of the feature column
    outputs:
        -
"""
def Save_CSV(file_name, x, y, header):
    import pandas as pd
    import numpy as np
    data = np.array([x,y]).transpose()
    df = pd.DataFrame(data, columns=[header, "Label"])
    df.to_csv(file_name)
    

"""
    Save_History(file_name, history)
    A function to automate the saving of training and testing history
    
    inputs:
        - file_name(string): The name of the file to store the history in
        - history (dictionary): The history dictionary from a TF history object
    outputs:
        -
"""
def Save_History(file_name, history):

    with open(file_name, 'wb') as file:
        pickle.dump(history, file)

            
"""
    Save_Model(file_name, model, base_name)
    A function to save ANN_Model objects to text files for later use.
    
    inputs:
        - file_name (string): Either a full path including the file name or a relative path to the models folder
        - model (ANN_Model): ANN_Model object to be saved
        - base_name (string): Model type spcifier used by model-generator.py
    outputs:
        -
"""
def Save_Model(file_name, model, base_name=None):
    try:
        file_name = file_name if base_name is None else file_name + base_name + "/" + model.get_attribute("name") + ".txt"            
        output = ""
        
        for key in model.report:
            value = str(model.report[key])
            value = value.replace("[","")
            value = value.replace("]","")
            output += str(key) + ":" + value + "\n"
        
        file = open(file_name, "w")
        file.write(output)
        file.close()
        
    except Exception as e:
        Print_Error("save model", e)
        

"""
    Save_Report(file_name, report)
    A function to automate the processing of class reports generated
    when building models or when training models. 
    
    inputs:
        - file_name (string): Specifies the save location
        - report (dictionary): Stores the informaion to save
    outputs:
        -
"""
def Save_Report(file_name, report):
    try:
        output = Build_Outputs(report)
        output += "\n"        
        file = open(file_name, "a")
        file.write(output)
        file.close()
    except Exception as e:
        Print_Error('save report', e)  
        
        
"""
    Save_Results(file_name, trainer_report, model_report, confusion_matrix)
    A function to automate writing Trainer, Model, and results to a text file.
    
    inputs:
        - file_name (string): Specifies the save location
        - trainer_report (dictionary): The trainers configuration report
        - model_report (dictionary): The models configuration report
        - confusion_matrix (numpy array): The 2d array of the models confusion matrix
    outputs:
        -
"""
def Save_Results(file_name, trainer_report, model_report, confusion_matrix):
    try:
        trainer_output = Build_Outputs(trainer_report)
        model_output = Build_Outputs(model_report)
        report_output = trainer_output + model_output
        report_output += "\n"
        
        report_output += Build_Confusion_Matrix_Output(confusion_matrix)
        
        file = open(file_name, "a")
        file.write(report_output)
        file.close()
        
        return True
        
    except Exception as e:
        Print_Error('save results', e)
        
        return False
        
        
"""
    Update_Confusion_Matrix(confusion_matrix, prediction, target, time_steps)
    A function to iterate over all predictions for one file and update a confusion matrix. The
    confusion matrix has the true values on the cols and the predictions on the rows. The 
    transpose of the input predictions is output for saving.
    
    inputs:
        - confusion_matrix (numpy array): The existing 2d confusion matrix
        - prediction (numpy array): An array of the models predictions
        - target (numpy array): An array of the data labels
        - time_steps (int): The offset between the labels and the predictions
    outputs:
        - (numpy array): The transpose of the input predictions array
"""
def Update_Confusion_Matrix(confusion_matrix, prediction, target, time_steps):
    predictions = []
    time_steps -= 1
    for index, label, in enumerate(prediction):
        row = np.argmax(label)
        col = target[index+time_steps]
        confusion_matrix[row,col] += 1
        predictions.append(row)

    return np.array(predictions).transpose()
        
        
"""
    validate_dir(directory)
    A function which checks to see if a directory exists.
    
inputs:
    - directory (string): Path to validate
outputs:
    - (boolean): True if the directory was created, False if it already exists 
"""
def Validate_Dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
        
    else:
        return False
    
    
"""
    validate_file(file)
    A wrapper function for pythons os.path.isfile() function
    
inputs:
    - file (string): Location of file to validate
outputs:
    - (boolean): True if the file exists, False otherwise
"""
def Validate_File(file):
    return os.path.isfile(file)
    
    
