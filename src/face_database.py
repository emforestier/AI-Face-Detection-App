import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class FaceDatabase:

    # The parameter is expected to be a string and the default parameter is called data
    def __init__(self, data_dir: str = "data"):
        # path object created where all face recognition information will be stores
        self.data_dir = Path(data_dir)

        # folder for image files
        self.faces_dir = self.data_dir / "faces"

        # file that maps faces to actual names
        self.index_file = self.data_dir / "index.json"

        # Creates directory if it doesn't exist already and ignores the function if it already exists
        self.faces_dir.mkdir(parents = True, exist_ok = True)

        # Creates an empty dictionary
        self.index = {}

        # attribute for internal use which reads the index json file and puts that information into the newly created dictionary
        self._load_index()

    def _load_index(self):
        # check if index file exists
        if self.index_file.exists():

            # open file to read & load json file
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)

        # If it does not exist then create an index to store info
        else:
            # Create new empty index
            self.index = {}
            self._save_index()

    def _save_index(self):

        # opens the file in write mode to overwrite index file
        with open(self.index_file, 'w') as f:

            # pushes updated data to hard drive
            json.dump(self.index, indent = 4, fp=f)

    # Made to convert the name into a file-safe name
    @staticmethod
    # expects a string and also returns a string
    def _sanitize_name(name: str) -> str:

        # Joins together symbols w/ underscores by checking if they are numbers or letters
        safe = "".join(c if c.isalnum() else "_" for c in name)

        # remove unnecessary underscores
        while "__" in safe:
            safe = safe.replace("__", "_")

        return safe.strip("_").lower()

    # Takes in the string of the users name, the image array, and the array embedding
    # Returns either true or false based on if the function executed successfully
    def save_face(self, name: str, face_image: np.ndarray, embedding: np.ndarray) -> bool:

        # turns name into readable file name
        safe_name = self._sanitize_name(name)

        # Combines the main faces folder w/ the persons name
        person_dir = self.faces_dir / safe_name

        # Makes a new folder if one does not already exist
        person_dir.mkdir(exist_ok=True)

        # Saves the image inside the persons folder and takes the image and encodes it into a file
        face_path = person_dir / "face.jpg"
        cv2.imwrite(str(face_path), face_image)

        # Saves the AIs data as a numpy vector
        embedding_path = person_dir / "embedding.npy"
        np.save(str(embedding_path), embedding)

        # Adds to the dictionary
        self.index[safe_name] = {
            "display_name": name,  # Original name
            "image_path": str(face_path),
            "embedding_path": str(embedding_path)
        }

        # Persist index to disk
        self._save_index()
        return True

    def get_all_faces(self) -> List[Dict]:
        # Creates an empty list of faces
        faces = []

        # Loops through each name and data linked to each other in the dictionary
        for safe_name, data in self.index.items():

            # Adds the data of each face to the list
            faces.append({
                "name": safe_name,
                "display_name": data.get("display_name", safe_name),
                "face_path": data["face_path"],
                "embedding_path": data["embedding_path"]
            })

        return faces

    def load_embedding(self, name: str) -> Optional[np.ndarray]:
        # Check if the name is already inside the index
        if name in self.index:

            embedding_path = self.index[name]["embedding_path"]

            # Check if the file exists
            if os.path.exists(embedding_path):
                # If it does then load the numpy array from the file
                return np.load(embedding_path)

        # if not return no embedding
        return None

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        # Creates empty dictionary
        embeddings = {}

        # Loops through the index dictionary of all saved faces
        for safe_name in self.index.keys():

            # Pulls out giant numpy array into the embedding
            embedding = self.load_embedding(safe_name)

            # If the embedding has valid information then it gets added to the index dictionary
            if embedding is not None:
                embeddings[safe_name] = embedding

        return embeddings

    def find_closest_match(self, query_embedding: np.ndarray,
                           threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        embeddings = self.load_all_embeddings()

        if not embeddings:
            return None

        # Initialize tracking variables
        best_match = None

        best_distance = float('inf')  # Start with infinity

        for name, saved_embedding in embeddings.items():

            # Distance = norm(query_embedding - saved_embedding)
            distance = np.linalg.norm(query_embedding - saved_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = name

        if best_distance <= threshold:
            # Get display name
            display_name = self.index[best_match].get("display_name", best_match)
            return (display_name, best_distance)

        return None

    def rename_face(self, old_name: str, new_name: str) -> bool:

        # If the old name is not already in the index then say the name cant be updated
        if old_name not in self.index:
            return False

        # Otherwise update the display name and save it to the index
        self.index[old_name]["display_name"] = new_name

        self._save_index()
        return True

    def delete_face(self, name: str) -> bool:
        # If name does not exist inside the index then say that the function did not need to be executed
        if name not in self.index:
            return False

        # Remove from filesystem
        person_dir = self.faces_dir / name
        if person_dir.exists():
            import shutil
            # Recursively delete directory using shutil.rmtree()
            shutil.rmtree(person_dir)

        # TODO: Remove from index using del keyword
        del self.index[name]

        self._save_index()
        return True

    def search_faces(self, query: str) -> List[Dict]:
        # Convert query to lowercase
        query_lower = query.lower()

        # Empty list which are supposed to hold the faces
        results = []

        # Search through all faces
        for face in self.get_all_faces():
            # Check if query is substring of display_name (case-insensitive)
            if query_lower in face["display_name"].lower():
                results.append(face)

        return results

