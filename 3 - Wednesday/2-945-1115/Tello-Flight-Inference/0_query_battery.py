"""
This Python script acts as a "handshake" to ensure that connection to the Tello Drone has been established.

I would recommend running this script before any other script to ensure that the Tello Drone is connected and ready to
receive commands.

A simple handshake is performed by querying the battery life of the drone and printing it to the console.
"""

# Import python-native modules
from djitellopy import Tello

if __name__ == "__main__":
    # Create a Tello() object and connect to the drone
    tello = Tello()
    tello.connect()

    # Query the battery life of the drone and print it to the console (displayed as a percentage)
    print(f"Battery Life: {tello.query_battery()}%")

    # Clean up
    tello.end()
