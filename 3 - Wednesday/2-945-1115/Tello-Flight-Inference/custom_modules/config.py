"""
This file contains all the parameters that are used in the training and testing of our Drone Obstacle CNN.
"""

##################################
# Image specific parameters
##################################
NUM_DUMMY_IMAGES = 1
NUM_CHANNELS = 3
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720
IMAGE_REFRESH_RATE = 0.2

##################################
# User specific parameters
##################################
# Hardware parameters
ACCELERATOR = "gpu"
DEVICES = [0]  # Default: [0] (i.e. use only the first GPU)
NUM_WORKERS = (
    1  # Can determine 'optimal' parameter from 'determine_optimal_num_workers.py'
)

# Dataset location
DATA_DIR = "sorted_data"  # Directory to store the sorted data

# Data sorting parameters (with usage of 2_binary_classification_automatic_labelling.py)
RAW_DATA_DIR = "raw_data"  # Directory storing the collected data from the drone
SORTED_DATA_DIR = "sorted_data"  # Directory to store the sorted data

# Logging/Saving parameters
TORCH_MODEL_DIRECTORY = (
    "torch_models/"  # Directory to store the PyTorch models after training
)
TORCH_MODEL_FILENAME = "drone_obstacle_cnn"  # Desired template filename for saving PyTorch models after training
TORCH_MODEL_FILENAME_EXT = (
    ".pt"  # Desired file extension for saving PyTorch models after training
)
TORCH_MODEL_FILENAME_LOAD = (
    "drone_obstacle_cnn_acc_0.8571.pt"  # Desired filename of PyTorch model to load
)
GOLDEN_MODEL_FILEPATH = "golden_model/drone_obstacle_cnn_acc_0.8571.pt"
INFERENCE_MODEL_FILENAME = "drone_obstacle_cnn_acc_0.8571.pt"  # Desired filename of PyTorch model to load for inference
CHECKPOINT_DIR = (
    "checkpoints"  # Directory to store the model checkpoints during training
)
TEST_AND_SAVE_MODEL_PT = True  # Set to True if you want to test the model after training and save the model.pt file
TEST_BEST_MODEL_CKPT = (
    True  # Set to True if you want to test the best model checkpoint after training
)
LOAD_AND_TEST = False  # Set to True if you want to load a model and test it
LOG_EVERY_N_STEPS = 1  # Set to 1 if you want to log every step during training
SAVE_IMAGES = False  # Set to True if you want to save images during training
AUTO_CONTROL = (
    False  # Set to True if you want to automatically control the drone on startup
)
RUN_INFERENCE = True  # Set to True if you want to start out running inference using the drone camera feed

# Parameters related to dataset "re-scrambling"
DATASET_FILE_EXT = ".jpg"  # File extension of the images in the dataset
SOURCE_FOLDER = "C:\\Users\\bcoburn1\OneDrive - University of Wyoming\\Desktop\\ML4HST_drone\\dev_code\\playground\\reference_CNN_lightning_model\\obstacle_dataset"  # Directory of the 'original' dataset (unsorted)
DESTINATION_FOLDER = "DRONE_OBSTACLES"  # Uncomment this if you want to augment the 'original' dataset directory
# DESTINATION_FOLDER = "DRONE_OBSTACLES_RESCRAMBLE" # Directory of the 're-scrambled' dataset (sorted)
CLASS_A_NAME = "BLOCKED"
CLASS_B_NAME = "UNBLOCKED"
SPLIT_RATIO = (
    0.8,
    0.1,
    0.1,
)  # Ratio of train, validation, and test datasets (Default: 80%, 10%, 10%)

# Parameters related to dataset collection
COLLECTION_BASE_DIR = "raw_data"  # Directory to store the collected data from the drone

##################################
# Hyperparameters
##################################
# Training parameters
BATCH_SIZE = 64
MAX_EPOCHS = 50
MIN_EPOCHS = 5
LEARNING_RATE = 0.001

# Early Stopping parameters
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.005

# Transformation parameters
SIZE_REDUCTION_FACTOR = 3

##################################
# Transformation parameters
##################################
RANDOM_ROTATION_DEGREES = 15
RANDOM_HORIZONTAL_FLIP_PERCENTAGE = 0.25
RANDOM_AFFINE_DEGREES = 15
RANDOM_VERTICAL_FLIP_PERCENTAGE = 0.25
RANDOM_CROP_PERCENTAGE = 0.1
