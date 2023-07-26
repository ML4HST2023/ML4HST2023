"""
This file contains functions that are used in the a "__main__" file, but are not
directly related to the main functionality of the file.
"""

# Import Python-native modules
import os
import shutil
import random
import math


def resize_image_dimensions(
    image_width, image_height, size_reduction_factor, verbose=True
):
    """
    This function takes in original image dimensions and returns the new
    image dimensions after applying a size reduction factor.
    """
    new_width = image_width / size_reduction_factor
    new_height = image_height / size_reduction_factor

    new_width = int(new_width)
    new_height = int(new_height)

    if verbose:
        print(f"========================================================")
        print(f"\tsize_reduction_factor: {size_reduction_factor}")
        print(f"Resizing image_width from {image_width} to {new_width}")
        print(f"Resizing image_height from {image_height} to {new_height}")
        print(f"========================================================")

    return new_width, new_height


def calculate_split_ratio(image_list, split_ratio):
    """
    Calculate the number of images for each split (in total)
    """
    split_ratio_sum = sum(split_ratio)
    if not math.isclose(split_ratio_sum, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Entries in 'split_ratio' do not add up to 1, add to {split_ratio_sum}, please modify SPLIT_RATIO in config.py"
        )

    total_images = len(image_list)
    train_ratio, val_ratio, test_ratio = split_ratio
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    return train_count, val_count, test_count


def split_images(
    destination_folder,
    class_a_name,
    class_a_folder,
    class_b_name,
    class_b_folder,
    split_ratio=(0.7, 0.2, 0.1),
    file_extension=".jpg",
):
    """
    Split the images into training, validation and test sets. The split is done randomly and depends on the split ratio.
    """
    # Create destination folders if they don't exist
    for folder in ["Train", "Val", "Test"]:
        os.makedirs(os.path.join(destination_folder, folder), exist_ok=True)

    # Collect the image files from ClassA and ClassB folders
    class_a_images = [
        file for file in os.listdir(class_a_folder) if file.endswith(file_extension)
    ]
    print("***************************************************************")
    print(f"Number of class {class_a_name} images: {len(class_a_images)}")
    class_b_images = [
        file for file in os.listdir(class_b_folder) if file.endswith(file_extension)
    ]
    print(f"Number of class {class_b_name} images: {len(class_b_images)}")
    print("***************************************************************")

    # Randomly shuffle the image files
    random.shuffle(class_a_images)
    random.shuffle(class_b_images)

    # Calculate the number of images for each split (in total, for class A and
    # for class B)
    total_train_count, total_val_count, total_test_count = calculate_split_ratio(
        class_a_images + class_b_images, split_ratio
    )
    class_a_train_count, class_a_val_count, class_a_test_count = calculate_split_ratio(
        class_a_images, split_ratio
    )
    class_b_train_count, class_b_val_count, class_b_test_count = calculate_split_ratio(
        class_b_images, split_ratio
    )

    print("----------------------------------------")
    print(f"TOTAL TRAIN COUNT: {total_train_count}")
    print(f"TOTAL VAL COUNT: {total_val_count}")
    print(f"TOTAL TEST COUNT: {total_test_count}")

    ##################################################################
    # Copy images to the destination folders based on the split count
    ##################################################################

    # Copy first for class A
    split_and_copy_images(
        class_a_images,
        class_a_train_count,
        class_a_val_count,
        class_a_test_count,
        destination_folder,
        class_a_name,
        class_a_folder,
    )

    # Copy next for class B
    split_and_copy_images(
        class_b_images,
        class_b_train_count,
        class_b_val_count,
        class_b_test_count,
        destination_folder,
        class_b_name,
        class_b_folder,
    )

    print("=============================")
    print("----> Splitting completed")
    print("=============================")


def split_and_copy_images(
    image_list,
    train_count,
    val_count,
    test_count,
    destination_folder,
    class_name,
    class_folder,
):
    """
    Copy images to 'Train', 'Val' and 'Test' folders based on the split counts
    """
    copy_images(
        image_list[:train_count],
        os.path.join(destination_folder, "Train", class_name),
        class_folder,
    )
    print(f"{class_name} train length: {len(image_list[:train_count])}")
    copy_images(
        image_list[train_count : train_count + val_count],
        os.path.join(destination_folder, "Val", class_name),
        class_folder,
    )
    print(
        f"{class_name} val length: {len(image_list[train_count:train_count + val_count])}"
    )
    copy_images(
        image_list[train_count + val_count : train_count + val_count + test_count],
        os.path.join(destination_folder, "Test", class_name),
        class_folder,
    )
    print(
        f"{class_name} test length: {len(image_list[train_count + val_count:train_count + val_count + test_count])}"
    )


def copy_images(image_list, destination_folder, source_folder):
    """
    Copy images from the source folder to the destination folder
    """
    for image in image_list:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copyfile(source_path, destination_path)


def delete_images_in_folder(folder_path, image_extension=".jpg"):
    """
    Delete all images in a provided folderpath
    """
    print(f"----------> Removing existing images from {folder_path}...")
    # Delete all images in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith(image_extension):
            os.remove(filepath)


def create_folder_if_not_exists(folder_path):
    """
    Create a folder if it does not exist, otherwise delete all images in the existing folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"---- Created folder: {folder_path}")
    else:
        print(f"---- Folder already exists: {folder_path}")
        delete_images_in_folder(folder_path)


def nice_print(string):
    """
    Method for a nice print of a '*' lined border!
    """
    border_length = len(string) + 4
    border = "*" * border_length

    print(border)
    print(f"* {string} *")
    print(border)
