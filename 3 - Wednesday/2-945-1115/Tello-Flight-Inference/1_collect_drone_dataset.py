"""
This Python script aims to connect to a Tello Drone and save images from its camera to a local directory.

The user still maintains control of the drone, but it is recommended to hold and rotate the drone while the
camera feed is being saved. This will allow for more distinct control while obtaining the dataset.
"""

# Import python-native modules
import time

# Import custom modules
from custom_modules.VideoStreamTello import VideoStreamTello
from custom_modules.supplemental_functions import nice_print

# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object and automatically start the video stream and user input polling
    ########################################################################
    tello_video_stream = VideoStreamTello(
        save_images=True,
        load_model=False,
        run_inference=False,
        auto_control=False,
    )
    ########################################################################

    # Enter our main execution loop (can only be exited via a user input
    # 'kill' or KeyboardInterrupt)
    while tello_video_stream.main_loop:
        try:
            tello_video_stream.poll_keystrokes()
        except KeyboardInterrupt:
            print(f"!!!Interrupted!!!")

            # Stop our main loop
            tello_video_stream.main_loop = False

            # Initiate the kill sequence
            tello_video_stream.kill_sequence()

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    nice_print(f"Done with main loop in {end_time} seconds...")
