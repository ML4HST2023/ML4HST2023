import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import numpy as np
import torch
import torchvision.transforms as transforms
from utils import load_model
from torchvision.models import resnet18, resnet50
import time
import random
import json

filename = 'bestModel_resnet50.tar'
use_gpu = True
model = resnet50(num_classes=1081)

load_model(model, filename=filename, use_gpu=use_gpu)

# Load the class mapping from the JSON file
with open('C:/Users/mspring6/Documents/ML4HST-2023/PlantID/plantnet_300K_images/plantnet300K_species_id_2_name_test.json') as f:
    class_mapping = json.load(f)

# Define the transformation for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a function to classify the image
def classify_image():
    timeNum = random.uniform(8.0, 12.0)
    result_label.config(text=f"Class: ")
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    
    # Load and preprocess the image
    image = Image.open(file_path)
    display_image(image)
    
    
    
    # Simulate classification progress
    classifying_label.config(text="Classifying...")
    window.update_idletasks()  # Update the GUI
    time.sleep(timeNum)
    
    image = transform(image).unsqueeze(0)
    
    # Predict the class of the image
    with torch.no_grad():
        prediction = model(image)
    #print(prediction)
    #print(prediction.shape)
    
    probabilities = torch.softmax(prediction, dim=1)
    #print(probabilities)
    # Get the predicted class label
    class_label = torch.argmax(probabilities).item()
    #print(class_label)
    
    # Retrieve the class name from the class mapping JSON
    class_name = class_mapping.get(str(class_label), "Unknown Class")
    
    # Update the GUI with the predicted class
    result_label.config(text=f"Class: {class_name}")
    
    # Reset the progress bar and classifying label
    classifying_label.config(text="")
    window.update_idletasks()  # Update the GUI

def display_image(image):
    # Resize the image to fit in the GUI
    image = image.resize((300, 300))
    
    # Convert the PIL Image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(image)
    
    # Update the image label in the GUI
    image_label.config(image=photo)
    image_label.image = photo

# Create the GUI window
window = tk.Tk()
window.title("Plant Identification")

# Set the window size
window.geometry("500x500")  # Adjust the width and height as desired

# Create a button to select an image
select_button = tk.Button(window, text="Select Image", command=classify_image)
select_button.pack(pady=10)


# Create a label for the "Classifying..." text
classifying_label = tk.Label(window, text="")
classifying_label.pack()

# Create a label to display the predicted class
result_label = tk.Label(window, text="Class: ")
result_label.pack()

# Create a label to display the inputted image
image_label = tk.Label(window)
image_label.pack()

# Start the GUI event loop
window.mainloop()
