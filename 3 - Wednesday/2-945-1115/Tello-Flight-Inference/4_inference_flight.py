"""
This Python script aims to load and utilize an inference model to classify images from a Tello drone's camera.
"""

# Import python-native modules
import time
import queue

# Import custom modules
from custom_modules.VideoStreamTello import VideoStreamTello
from custom_modules.supplemental_functions import nice_print
from custom_modules.CommandPopup import CommandPopup
from custom_modules.command_dict import command_dict
from custom_modules import config

# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object and automatically start the video stream and user input polling
    ########################################################################
    # Ensure that config.SAVE_IMAGES is set your preference (True/False) PRIOR to running this script
    # Ensure that config.AUTO_CONTROL is set your preference (True/False) PRIOR to running this script
    tello_video_stream = VideoStreamTello(
        save_images=config.SAVE_IMAGES,
        auto_control=config.AUTO_CONTROL,
        run_inference=config.RUN_INFERENCE,
        inference_model_filepath=config.TORCH_MODEL_DIRECTORY
        + config.INFERENCE_MODEL_FILENAME,
    )
    ########################################################################

    # Create a queue to hold commands from the command popup (to handoff to the VideoStreamTello object)
    command_queue = queue.Queue()
    command_popup = CommandPopup(command_queue=command_queue, command_dict=command_dict)

    # Enter our main execution loop (can only be exited via a user input
    # 'kill' or KeyboardInterrupt)
    while tello_video_stream.main_loop:
        # Update our command popup window
        command_popup.window.update()

        try:
            # Poll our command queue for new commands
            command = command_queue.get_nowait()
            tello_video_stream.get_button_command(command)
        except queue.Empty:
            # If there are no commands, pass
            pass
        except KeyboardInterrupt:
            # If we receive a KeyboardInterrupt, print a message, initiate the kill sequence, and join our threads
            print(f"!!!Interrupted!!!")

            # Stop our main loop
            tello_video_stream.main_loop = False

            # Initiate the kill sequence
            tello_video_stream.kill_sequence()

            # Join our running threads
            tello_video_stream.video_stream_t.join()
            tello_video_stream.image_save_t.join()
            tello_video_stream.inference_t.join()

    # Destroy command popup window
    command_popup.window.destroy()

    if config.SAVE_IMAGES:
        # Print our 'saving images' information
        nice_print(
            f"Wrote {tello_video_stream.num_images_written} images in {tello_video_stream.time_to_save_imgs_start} seconds"
        )

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    nice_print(f"Ran Tello code for {end_time} seconds...")
