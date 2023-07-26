# Import Python-native modules
import datetime
from djitellopy import Tello
import cv2
import time
from threading import Thread
import os
import torch
from torchvision import transforms
from PIL import Image
import random

# Import custom modules
from . import config
from .CNN_lightning import CNN_lightning


class VideoStreamTello(object):
    def __init__(
        self,
        unit_dp=30,
        window_name="Drone Camera",
        auto_control=True,
        run_inference=True,
        save_images=True,
        load_model=True,
        inference_model_filepath=config.TORCH_MODEL_DIRECTORY
        + config.INFERENCE_MODEL_FILENAME,
    ):
        # Ensure that a valid model filepath is provided
        if inference_model_filepath is None:
            raise ValueError(
                "inference_model cannot be None, please include a filepath to the inference model"
            )

        # Resize image dimensions prior to network initialization
        image_width, image_height = self.resize_image_dimensions(
            image_width=config.IMAGE_WIDTH,
            image_height=config.IMAGE_HEIGHT,
            size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
        )

        if load_model:
            # Load our inference model
            self.inference_model = CNN_lightning(
                num_dummy_images=config.NUM_DUMMY_IMAGES,
                num_channels=config.NUM_CHANNELS,
                image_width=image_width,
                image_height=image_height,
            )
            print(f"Load model: {inference_model_filepath}")
            self.inference_model.load_state_dict(torch.load(inference_model_filepath))
            self.inference_model.eval()

        # Establish Tello() object
        self.tello = Tello()

        # Connect to the tello
        self.tello.connect()

        # Query and print out the battery percentage
        self.query_battery()

        # Create a command dictionary where we expect a single keyword
        self.state_dictionary = {
            "kill": self.kill_sequence,
            "land": self.initiate_land,
            "takeoff": self.initiate_takeoff,
            "diag": self.diag,
            "auto_nav": self.auto_nav,
            "stop_auto_nav": self.stop_auto_nav,
            "inference": self.inference,
            "stop_inference": self.stop_inference,
        }

        # Create a command dictionary where we expect a single keyword AND a
        # parameter
        self.movement_dictionary = {
            "w": self.tello.move_forward,
            "s": self.tello.move_back,
            "a": self.tello.move_left,
            "d": self.tello.move_right,
            "e": self.tello.rotate_clockwise,
            "q": self.tello.rotate_counter_clockwise,
            "r": self.tello.move_up,
            "f": self.tello.move_down,
        }

        # Turn on the video stream from the tello
        self.tello.streamon()

        # Get the current video feed frame and convert into an image (for
        # display purposes)
        self.camera_frame = self.tello.get_frame_read()

        self.img = (
            self.camera_frame.frame
        )  # Get the current frame (instantiate as some image)
        self.most_recent_image = (
            self.img
        )  # Instantiate the most recent image as the current image

        self.num_images_written = 0  # Initialize the number of images written to 0
        self.time_to_save_imgs_start = (
            0  # Initialize the start time for saving images to 0
        )
        self.time_to_save_imgs_end = 0  # Initialize the end time for saving images to 0

        # Establish object attributes
        self.unit_dp = unit_dp  # Length of spatial displacement
        self.window_name = window_name  # Name of the video stream popup window
        ################################################
        self.image_refresh_rate = (
            config.IMAGE_REFRESH_RATE
        )  # How often to refresh the video stream
        ################################################
        self.landed = (
            True  # Boolean flag to determine whether the tello is on the ground
        )
        self.stream = (
            True  # Boolean flag to determine whether the Tello is streaming video
        )
        self.popup = True  # Boolean flag to determine whether the video stream popup window is open
        self.main_loop = (
            True  # Boolean flag to determine whether the main loop is running
        )

        self.save = save_images  # Boolean flag to determine whether to save images from the camera feed
        self.run_inference = run_inference  # Boolean flag to determine whether to run inference on the camera feed
        self.auto_control = (
            auto_control  # Boolean flag to determine whether to auto control the drone
        )

        # This is the value that will be updated by the inference model
        # (0 = blocked, 1 = unblocked)
        self.blocked_or_unblocked = (
            5.5555  # Default to a 'weird' value (so we can tell if it has been updated)
        )

        self.inference_threshold = 0.5  # Threshold for determining whether the inference model predicts blocked or unblocked
        self.random_threshold = (
            0.5  # Threshold for whether the drone should randomly turn left or right
        )
        self.forward_speed = 15  # Forward speed for the drone
        self.turn_speed = 20  # Turn speed for the drone
        self.turn_wait = 2

        # Setting some attributes which will be necessary for saving frames
        # from the camera feed
        self.base_directory = (
            config.COLLECTION_BASE_DIR
        )  # Base directory for saving images (should be 'raw_data')
        self.image_extension = (
            config.DATASET_FILE_EXT
        )  # Image extension (should be '.jpg')

        # These attributes are also necessary for saving frames from the camera
        # feed, but will be altered from other methods
        self.existing_runs = (
            None  # List of existing runs (directories) in the base directory
        )
        self.run_number = None  # Run number (directory number) for the current run
        self.directory_name = None  # Name of the current run directory
        self.timestamp = (
            None  # Timestamp for the current image (to be used as the filename)
        )
        self.filename = None  # Filename for the current image (utilizing timestamp and image extension)
        self.image_path = None  # Full path to the current image (utilizing the base directory, run directory, and filename)

        # Image save based methods
        if self.save:
            self.instantiate_base_directory()
            self.check_for_run_directories()

        # Threading is necessary to concurrently display the live video feed, get keystrokes from user, and save
        # images from the camera feed
        self.video_stream_t = Thread(target=self.update_frame, args=())
        self.video_stream_t.start()

        self.image_save_t = Thread(target=self.image_save, args=())
        self.image_save_t.start()

        self.inference_t = Thread(target=self.run_through_inference, args=())
        self.inference_t.start()

    def auto_nav(self):
        """
        Method to initiate the auto navigation / control sequence (i.e. run inference, and move the tello based on the
        inference output)
        """

        # If the tello is on the ground, takeoff
        if self.landed:
            self.initiate_takeoff()
            time.sleep(
                2
            )  # Wait a few seconds after takeoff to allow the tello to stabilize

        # Change the Boolean flag to run inference to True (read by the inference thread)
        self.run_inference = True
        self.auto_control = True

    def inference(self):
        """
        Method to initiate the auto navigation / control sequence (i.e. run inference, and move the tello based on the
        inference output)
        """
        # Change the Boolean flag to run inference to True (read by the inference thread)
        self.run_inference = True
        self.auto_control = False

    def stop_auto_nav(self):
        """
        Method to stop the auto navigation / control sequence (i.e. stop inference, and stop the tello)
        """

        if self.auto_control:
            # Change the Boolean flag to run inference to False (read by the inference thread)
            # self.run_inference = False
            self.auto_control = False

            if not self.landed:
                # Stop the tello in place
                self.tello.send_rc_control(
                    left_right_velocity=0,
                    forward_backward_velocity=0,
                    up_down_velocity=0,
                    yaw_velocity=0,
                )

    def stop_inference(self):
        """
        Method to stop the auto navigation / control sequence (i.e. stop inference, and stop the tello)
        """
        if self.run_inference:
            # Change the Boolean flag to run inference to False (read by the inference thread)
            self.run_inference = False
            self.auto_control = False

            if not self.landed:
                # Stop the tello in place
                self.tello.send_rc_control(
                    left_right_velocity=0,
                    forward_backward_velocity=0,
                    up_down_velocity=0,
                    yaw_velocity=0,
                )

    def run_through_inference(self):
        """
        Method to collect the most recently saved image from the camera feed, and feed it to the inference model
        """
        while self.run_inference:
            try:
                # Resize the image to the dimensions expected by the inference model
                image_width, image_height = self.resize_image_dimensions(
                    image_width=config.IMAGE_WIDTH,
                    image_height=config.IMAGE_HEIGHT,
                    size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
                )

                # Create the resize transform (passing in 'antialias' parameter to suppress warning)
                resize_transform = transforms.Resize(
                    (image_width, image_height), antialias=True
                )

                # Store the most recent image from the camera feed
                inference_image = self.most_recent_image

                # Convert the image to a PIL image
                inference_image = Image.fromarray(inference_image)

                # Convert the image to a tensor
                inference_image = transforms.ToTensor()(inference_image)

                # Add a batch dimension to the image (Model expects 4D input - originally a 3D input)
                inference_image = inference_image.unsqueeze(0)

                # Apply the transform to the image prior to feeding it to the inference model
                resized_image = resize_transform(inference_image)

                # Feed the image to the inference model
                blocked_or_unblocked = self.inference_model(resized_image)
                blocked_or_unblocked = round(blocked_or_unblocked.item(), 4)

                # Update the blocked_or_unblocked attribute
                self.blocked_or_unblocked = blocked_or_unblocked

                if self.auto_control:
                    # Automatically control the tello based on the inference output (self.blocked_or_unblocked)
                    self.auto_control_tello()

                # Wait for a bit before trying again
                time.sleep(self.image_refresh_rate)

            except KeyboardInterrupt:
                self.run_inference = False
                self.auto_control = False
                break

    @staticmethod
    def resize_image_dimensions(
        image_width, image_height, size_reduction_factor, verbose=False
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

    def instantiate_base_directory(self):
        """
        Method to establish where you want your 'run' directories to be stored (for camera feed images)
        """

        if not os.path.exists(self.base_directory):
            self.nice_print(f'Creating "{self.base_directory}" directory')
            os.makedirs(self.base_directory)

    def check_for_run_directories(self):
        """
        Method to create a directory corresponding to the current run within a 'base_directory'
        """

        # Check for existing run directories
        self.existing_runs = [
            dir_name
            for dir_name in os.listdir(self.base_directory)
            if dir_name.startswith("run")
        ]

        # Determine the run number
        self.run_number = len(self.existing_runs) + 1
        self.nice_print(f"Number of existing run directories: {self.run_number}")

        # Establish the new directory name
        self.directory_name = f"run{self.run_number:03}"

        # Check if the directory already exists
        while self.directory_name in self.existing_runs:
            self.nice_print(
                f'"{self.directory_name}" already exists. Incrementing run number...'
            )
            self.run_number += 1
            self.directory_name = f"run{self.run_number:03}"

        # Create the new directory
        self.nice_print(f"Creating {self.directory_name}...")
        os.makedirs(
            os.path.join(self.base_directory, self.directory_name), exist_ok=True
        )

    def image_save(self):
        """
        Method to save images from the Tello Camera feed
        """
        self.time_to_save_imgs_start = time.time()

        while self.save:
            try:
                # Create timestamp which will be used for the saved image
                # filename (creates timestamp down to the millisecond)
                self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

                # Generate the filename using the timestamp
                self.filename = self.timestamp + self.image_extension

                # Set the path for the new image
                self.image_path = os.path.join(
                    self.base_directory, self.directory_name, self.filename
                )

                # Save the image in the new directory
                cv2.imwrite(self.image_path, self.img)
                self.num_images_written += 1

                ###############################################################
                # Adjust this sleep parameter to change the number of images saved per second
                # Ex: time.sleep(0.1) will (roughly) save 10 images per second
                ###############################################################
                time.sleep(self.image_refresh_rate)
                ###############################################################

            except KeyboardInterrupt:
                break

    @staticmethod
    def nice_print(string):
        """
        Method for a nice print of a '*' lined border!
        """
        border_length = len(string) + 4
        border = "*" * border_length

        print(border)
        print(f"* {string} *")
        print(border)

    def query_battery(self):
        """
        Method to query and print the current battery percentage of the tello
        """
        print(f"Battery Life: {self.tello.query_battery()}%")

    def update_frame(self):
        """
        Method to update the live video feed from the tello (thread-based)
        """
        while self.stream:
            try:
                # Get the current image frame from the video feed and display
                # in a popup window
                self.camera_frame = self.tello.get_frame_read()
                self.img = self.camera_frame.frame
                self.most_recent_image = self.img
                self.window_name = "Drone Camera"

                # Display the blocked/unblocked probability to the popup window (only if inference is running)
                # Otherwise the blocked/unblocked probability will be displayed/saved during image collection
                if self.run_inference:
                    text = "p(blocked_or_unblocked):" + str(self.blocked_or_unblocked)
                    cv2.putText(
                        self.img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                # Display the image in a popup window
                cv2.imshow(self.window_name, self.img)

                # 'waitKey' is necessary to properly display a cv2 popup window
                cv2.waitKey(1)
            except KeyboardInterrupt:
                break

    def initiate_land(self):
        """
        Method to land the tello and set the 'landed' attribute to True
        """
        self.tello.land()
        self.landed = True

    def initiate_takeoff(self):
        """
        Method to have the tello takeoff and set the 'landed' attribute to False
        """
        self.tello.takeoff()
        self.landed = False

    def poll_keystrokes(self):
        """
        Method to capture user input (for tello-based movements)
        """
        command = input("Enter command (and argument(s)): ")

        # Split the input into separate strings
        command_list = command.split()

        # Check whether the input should be decoded using 'state_dictionary' (single keyword)
        # or using 'movement_dictionary' (single keyword and parameter)
        if len(command_list) == 1:
            try:
                requested_command = self.state_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__}")
                requested_command()
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        elif len(command_list) == 2:
            try:
                requested_command = self.movement_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__} {command_list[1]}")
                requested_command(int(command_list[1]))
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        else:
            print(f"Unrecognized inputs: {command_list}")

        # Get remaining battery percentage after each command has been sent
        self.query_battery()

    def get_button_command(self, command):
        """
        Method to capture user input (for tello-based movements)
        """

        # Split the input into separate strings
        command_list = command.split()
        self.nice_print(f"Received command: {command_list}")

        # Check whether the input should be decoded using 'state_dictionary' (single keyword)
        # or using 'movement_dictionary' (single keyword and parameter)
        if len(command_list) == 1:
            try:
                requested_command = self.state_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__}")
                requested_command()
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        elif len(command_list) == 2:
            try:
                requested_command = self.movement_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__} {command_list[1]}")
                requested_command(int(command_list[1]))
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        else:
            print(f"Unrecognized inputs: {command_list}")

        # Get remaining battery percentage after each command has been sent
        self.query_battery()

    def diag(self):
        """
        Method to print out the current state of various Boolean values
        """
        print(f"stream: {self.stream}")
        print(f"landed: {self.landed}")
        print(f"main_loop: {self.main_loop}")
        print(f"save: {self.save}")
        print(f"run_inference: {self.run_inference}")
        print(f"auto_control: {self.auto_control}")

    def auto_control_tello(self):
        """
        Method to automatically navigate the drone based on the blocked_or_unblocked
        probability
        """
        # If the probability of being blocked or unblocked is greater than
        # the threshold, then the drone will move forward
        if self.blocked_or_unblocked < self.inference_threshold:  # Blocked
            """
            If the drone is blocked, then it will randomly turn left or right until it is unblocked.

            Establish a random number between 0 and 1. If the number is less than self.random_threshold (default: 0.5),
            then the drone will move left. If the number is greater than self.random_threshold, then the drone will
            move right.
            """
            print(f"Detecting blocked with: {self.blocked_or_unblocked}")

            random_number = random.random()
            if random_number < self.random_threshold:  # Turn left
                print(f"\tTurning left...")
                self.tello.send_rc_control(
                    left_right_velocity=0,
                    forward_backward_velocity=0,
                    up_down_velocity=0,
                    yaw_velocity=-self.turn_speed,
                )
                time.sleep(self.turn_wait)
            else:  # Turn right
                print(f"\tTurning right...")
                self.tello.send_rc_control(
                    left_right_velocity=0,
                    forward_backward_velocity=0,
                    up_down_velocity=0,
                    yaw_velocity=self.turn_speed,
                )
                time.sleep(self.turn_wait)

        else:  # Unblocked
            # If the probability of being blocked or unblocked is less than the threshold, then the drone will move
            # forward until it is blocked
            self.tello.send_rc_control(
                left_right_velocity=0,
                forward_backward_velocity=self.forward_speed,
                up_down_velocity=0,
                yaw_velocity=0,
            )
            # time.sleep(1)

    def kill_sequence(self):
        """
        Method to completely stop all Tello operations other than the connection
        """

        print(f"---- killing main loop...")
        if self.main_loop:
            self.main_loop = False

        print(f"---- killing stream...")
        if self.stream:
            self.tello.streamoff()
            self.stream = False

        print(f"---- killing landing...")
        if not self.landed:
            self.tello.land()
            self.landed = True

        print(f"---- killing popups...")
        if self.popup:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.popup = False

        print(f"---- killing save state...")
        if self.save:
            self.save = False

            self.time_to_save_imgs_end = time.time() - self.time_to_save_imgs_start

            # Display the number of images written
            self.nice_print(
                f"Wrote {self.num_images_written} images in {self.time_to_save_imgs_end} seconds"
            )

        print(f"---- killing inference state...")
        if self.run_inference:
            self.run_inference = False

        print(f"---- killing auto control state...")
        if self.auto_control:
            self.auto_control = False

        # Join our running threads
        print(f"---- Joining threads...")
        self.video_stream_t.join()
        self.image_save_t.join()
        self.inference_t.join()

        print(f"---- Turning off stream...")
        self.tello.streamoff()

        # print(f"---- Closing connection...")
        # self.tello.end()

    @staticmethod
    def _inline_print(string, verbose=True):
        """
        Method to print a string inline
        """
        if verbose:
            # Clear the line
            print_string = "\b" * len(string)
            print(print_string, end="", flush=True)
            print(string, end="", flush=True)
