"""
Python script to pass a directory of images to a trained model and automatically sort them into their respective
class directories.

Depending on the 'split_ratio' passed in by the user, this script will split the images into a training, validation,
and test directories (to align with PyTorch's ImageFolder class).

The split is done randomly and depends on the split ratio.

A progress bar is used to show the progress of our image classification.
"""

# Import Python-native modules
import os
import shutil
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm

# Import custom modules
from custom_modules.CNN_lightning import CNN_lightning
from custom_modules.supplemental_functions import resize_image_dimensions
from custom_modules import config


# Function to load a PyTorch model
def load_model(model_path, image_width, image_height):
    # Create an instance of our model

    ##################################################
    # Change the model instantiation to match the model you want to load...
    model = CNN_lightning(
        num_dummy_images=config.NUM_DUMMY_IMAGES,
        num_channels=config.NUM_CHANNELS,
        image_width=image_width,
        image_height=image_height,
        verbose=False,
    )
    ##################################################

    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


# Function to classify an image using a loaded model
def classify_image(image_path, model, image_width, image_height):
    # Create a transform to apply to the image where we resize it to the model's input dimensions
    # and convert it to a tensor
    transform = transforms.Compose(
        [
            transforms.Resize((image_width, image_height), antialias=True),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # Load the image and apply the transform (unsqueeze to add a batch dimension - previously a 3D tensor, now 4D)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
        # print(f'model_output: {output}')

        # Convert the output to a class prediction (and associate it with the class label))
        if output.item() > 0.5:
            predicted_class = 1  # config.CLASS_A_NAME (I think)
        else:
            predicted_class = 0  # config.CLASS_B_NAME (I think)

    # print(f'\predicted_class: {predicted_class}')
    return predicted_class


# Function to organize images into separate class folders with a progress bar
def organize_images_with_progress(
    source_dir, destination_dir, model, split_ratio, image_width, image_height
):
    classes = [config.CLASS_A_NAME, config.CLASS_B_NAME]

    # Clear the destination directory if it exists
    if os.path.exists(destination_dir):
        print(
            "\n\n================================================================================="
        )
        print(f"Destination directory exists: '{destination_dir}', deleting images...")
        shutil.rmtree(destination_dir)
    else:
        print(
            "\n\n================================================================================="
        )
        print(
            f"Destination directory does not exist: '{destination_dir}', creating directory..."
        )

    # Create destination directories where each class is a subdirectory of the Test, Train, and Val parent-directories
    os.makedirs(destination_dir, exist_ok=True)
    for split in ["Train", "Test", "Val"]:
        split_dir = os.path.join(destination_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

    # Get a list of image files to process
    image_files = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_files.append(os.path.join(root, filename))

    print(f"********** Number of images to process: {len(image_files)} **********")

    # Create a tqdm progress bar
    pbar = tqdm(total=len(image_files), desc="Classifying Images", unit="image")

    # Store the count of Test, Train, and Val images for each class
    count = {
        class_name: {"Test": 0, "Train": 0, "Val": 0, "Total": 0}
        for class_name in classes
    }

    # Iterate through the image files
    for image_path in image_files:
        class_index = classify_image(image_path, model, image_width, image_height)

        # Split the image into Train, Test, and Val directories based on the split ratio
        split_probabilities = torch.Tensor(split_ratio)
        split = torch.multinomial(split_probabilities, 1).item()
        splits = ["Train", "Test", "Val"]
        split_name = splits[split]

        # Copy the image to the corresponding class and split directory
        destination_class_dir = os.path.join(
            destination_dir, split_name, classes[class_index]
        )
        os.makedirs(destination_class_dir, exist_ok=True)
        shutil.copy(image_path, destination_class_dir)

        # Update the count for the split
        count[classes[class_index]][split_name] += 1
        # Update the total count
        count[classes[class_index]]["Total"] += 1

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Print the summary
    for class_name in classes:
        class_str = f"Class: {class_name}"
        max_length = len(class_str) + 2
        print(f"+{'=' * max_length}+")
        print(f"| {class_str} |")
        print(f"+{'-' * max_length}+")
        for split in ["Test", "Train", "Val"]:
            count_str = str(count[class_name][split])
            print(f"| {split}: {count_str:<{max_length - len(split + count_str)}}|")
        print(f"+{'-' * max_length}+")
        total_str = str(count[class_name]["Total"])
        print(f"| Total: {total_str:<{max_length - len(total_str + 'Total')}}|")
        print(f"+{'=' * max_length}+")
        print()


if __name__ == "__main__":
    # Set variables
    model_path = config.GOLDEN_MODEL_FILEPATH
    source_dir = config.RAW_DATA_DIR
    destination_dir = config.SORTED_DATA_DIR
    split_ratio = (
        config.SPLIT_RATIO
    )  # 80% Train, 10% Test, 10% Val (or whatever is in config.py)

    # Set image dimensions
    image_width, image_height = resize_image_dimensions(
        image_width=config.IMAGE_WIDTH,
        image_height=config.IMAGE_HEIGHT,
        size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
        verbose=False,
    )

    # Load the model
    model = load_model(model_path, image_width, image_height)

    # Organize the images with a progress bar display
    organize_images_with_progress(
        source_dir, destination_dir, model, split_ratio, image_width, image_height
    )
