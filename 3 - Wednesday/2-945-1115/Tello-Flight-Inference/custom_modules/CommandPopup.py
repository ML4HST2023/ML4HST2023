"""
This Python script houses the object which will be used to create a popup window for the user to click command buttons
"""

# Import python-native modules
import tkinter as tk

# Import custom modules
from .supplemental_functions import nice_print


class CommandPopup(object):
    """
    This class is used to create a popup window for the user to click command buttons
    """

    def __init__(self, command_queue, command_dict):
        """
        Initialize the CommandPopup() object

        :param command_queue: (queue) A queue object which will be used to send commands to the Tello drone
        """

        # Save the command queue
        self.command_queue = command_queue

        # Create the main window
        self.window = tk.Tk()

        # Set the window title
        self.window.title("Command Buttons")

        # Function to create and handle button click events
        def create_button(text, command):
            """
            This function will create a button with the given text and command

            :param text: (str) The text to be displayed on the button
            :param command: (str) The command to be sent to the Tello drone
            """

            # Create the button
            button = tk.Button(
                self.window, text=text, command=lambda: self.send_command(command)
            )

            # Pack the button
            button.pack(pady=10)

        # Create the buttons based off of the command_dict
        for key, value in command_dict.items():
            create_button(key, value)

    def send_command(self, command):
        """
        This function will send the given command to the Tello drone by putting it in the command queue

        :param command: (str) The command to be sent to the Tello drone
        """

        # Print the command we are sending
        nice_print(f"Sending command: {command}")

        # Send the command to the command queue
        self.command_queue.put(command)

    def start(self):
        """
        This function will start the Tkinter event loop

        For now, this method is unused as the GUI is being updated through window.update() instead of the mainloop()
        method.
        """

        # Start the Tkinter event loop
        self.window.mainloop()
