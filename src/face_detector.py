import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from typing import List, Tuple, Optional

class FaceDetector:

    # Uses the AI to get the models that specialize in tracking face movements
    # the mp_face_mesh instance now references the AI face module where you can create a face object, access drawing functions, and make connections between landmarks on each face
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        # Creates a face object with default attributes
        self.face_mesh = self.mp_face_mesh.FaceMesh(

            # Static image mode has an on and off mode, (True) means that the AI will treat every frame like a brand-new photo, but it's slow because it starts the land marking from scratch every frame. (false) This mode finds the face in frame one and only tracks it in frame 2 which is a lot faster
            static_image_mode = False,

            # Tells the AI the max number of faces to track at once
            max_num_faces = 10,

            # The AIs confidence that it is a certain face before they associate it with a specific object
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )


        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    # The expected input is a numPy array of numbers to represent horizontal and vertical  which is the image
    # The expected output is a tuple of a giant list of face coordinates and a list of extra information about the face
    def detect_faces(self, image: np.ndarray) -> Tuple[List, List]:

        rbg_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # model hog is for speed and change to cnn for accuracy
        face_locations = face_recognition.face_locations(rbg_image, model="hog")
        results = self.face_mesh.face_mesh.process()
        return face_locations, results

    # This function is to turn the face into a mathematical representation of the specific face object
    # Input the image and a tuple of the face coordinates
    # The output is expected to be a numpy array however it can also be None
    def get_face_encoding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Uses the image and the list of face locations to retrieve the face encodings
        encodings = face_recognition.face_encodings(rgb_image, face_location)

        if encodings:
            return encodings[0]
        return None

    # In this function it accepts a list of tuples containing multiple face locations based on the max number of faces allowed be tracked at once
    def draw_face_boxes(self, image: np.ndarray, face_locations: List) -> np.ndarray:
        # Creates an extra copy of the face box to avoid altering the original face box
        output = image.copy()

        # Loops through the tuples in the list of face locations and assigns each of the faces coordinates to points in a rectangle to create boxes around their faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)

        return output

    def draw_face_landmarks(self, image: np.ndarray, face_mesh_results) -> np.ndarray:
        # Creates copy
        output = image.copy()

        # If faces were detected
        if face_mesh_results.multi_face_landmarks:

            # Loop through all the faces: tesselation being the topology/surface volume of the face and contours drawing lines between various expressive features
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image = output,
                    landmark_list = face_landmarks,
                    connections = self.mp_face_mesh.FACEMESHTESSELATION,
                    landmark_drawing_spec = None,
                    connectiong_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                self.mp_drawing.draw_landmarks(
                    image = output,
                    landmark_list = face_landmarks,
                    connections = self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # New numpy array where the pixels have been detected, tesselation and contours have been applied to hypothetical face objects
        return output

    # Takes in a numpy array (frame image), the face location in the form of a tuple, and the padding
    # returns numpy array w/ new padding
    def crop_face(self, image: np.ndarray, face_location: Tuple, padding: int = 20) -> np.ndarray:
        # unpacks face locations into 4 variables
        top, right, bottom, left = face_location

        # unpacks height and width
        height, width = image.shape[:2]

        # subtract or add padding to adjust the edges either towards the top left or towards the bottom right
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)

        return image[top:bottom, left:right]

    def cleanup(self):
        self.face_mesh.close()