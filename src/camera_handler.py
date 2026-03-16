# File which is responsible for handling all webcam interactions via OpenCV VideoCapture API

import cv2
import numpy as np

# Allows the program to handle a case where none may be the value of a variable
from typing import Optional

class CameraHandler:

    # int = 0 being a TypeHint to declare a specific type for the Python variable
    def __init__(self, camera_index: int = 0):

        self.camera_index = camera_index
        self.capture = None
        self.camera_running = False

    # Function in order to start up the camera
    # -> bool : Indicates the return type since Python doesn't require return types in function declaration
    def start(self) -> bool:

        # Checking if camera is already running
        if self.capture is not None:
            return True

        # VideoCapture is a class within the OpenCV library
        self.capture = cv2.VideoCapture(self.camera_index)

        # Checking is the capture/selected camera is open and if not then return False
        if not self.capture.isOpened():
            self.capture = None
            return False

        # If none of the previous if statements return false then return true to show the camera is fully working and ready for use
        self.camera_running = True
        return True

    # Function to close down the camera
    def stop(self):

        self.camera_running = False

        if self.capture is not None:
            # release function necessary to telling the OS that the program is finished using the camera
            self.capture.release()
            self.capture = None

    # Purpose of the read_frame function is to turn the raw data of a frame into a NumPy array so that Python can actually use the data from the properties of the frame

    # Return type specified as optional to say that you can also return type None
    # The array that can also be returned is the frame's height, width, and color channels
    def read_frame(self) -> Optional[np.ndarray]:

        # If statement to return a frame of type None if the camera is not specified and not actively running
        if self.capture is None or not self.camera_running:
            return None

        # read() function returns a success boolean value and the frame as a numPy array in the form of a tuple which is unpacked into ret and frame
        ret, frame = self.capture.read()

        # So, if the frame is successfully unpacked from the capture the frame is returned
        if ret:
            return frame
        return None

    # Returns a tuple of width and height that can be later unpacked
    def get_frame_size(self) -> tuple:
        # If the camera is not found, return default dimensions
        if self.capture is None:
            return (640, 480)

        # get function is reaching into the camera's metadata to pull out info
        # cv2 arguments in the get function get the type of information needed (in this case the ID number of the horizontal and vertical pixel counts)
        # the int that surrounds the function changes the float number of the pixels to a whole number so that Python can efficiently use it
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (width, height)









