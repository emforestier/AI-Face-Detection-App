
import sys
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QScrollArea, QFileDialog,
    QStatusBar, QToolBar, QMessageBox, QInputDialog, QFrame
)

from PySide6.QtCore import Qt, QTimer, Signal, QSize

from PySide6.QtGui import QImage, QPixmap, QAction

from face_detector import FaceDetector
from face_database import FaceDatabase
from camera_handler import CameraHandler


class FaceItemWidget(QFrame):

    # Defining the signal which sends two strings, the ID and the new name to the main window so that the name can be successfully updated
    rename_requested = Signal(str, str)

    def __init__(self, name: str, display_name: str, face_path: str):
        super().__init__()

        # Initializes class attributes
        self.name = name
        self.display_name = display_name
        self.face_path = face_path

        # Create horizontal layout so that the face thumbnail is on the left, name in the middle, and editing buttons on the right
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(64, 64)
        self.thumbnail.setScaledContents(True)
        self.load_thumbnail()
        layout.addWidget(self.thumbnail)

        # TODO: Create name label with display_name
        self.name_label = QLabel(display_name)
        self.name_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.name_label, 1)  # Stretch factor


        rename_btn = QPushButton("✏️: RENAME")
        rename_btn.setFixedSize(30, 30)
        rename_btn.clicked.connect(self.on_rename_clicked)
        layout.addWidget(rename_btn)

        self.setLayout(layout)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)

    def load_thumbnail(self):

        # Checking if the file exists before they open it
        if Path(self.face_path).exists():

            image = cv2.imread(self.face_path)

            # If frame is shown
            if image is not None:
                # Resize the image to keep the app fast
                image = cv2.resize(image, (64, 64))

                # Convert BGR to RGB for Qt
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                # Tells the Qt library to not copy the image and look at the original memory address to reference the original to make edits
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                                   QImage.Format_RGB888)

                # Ensures that it doesn't lag by converting to pixmap
                self.thumbnail.setPixmap(QPixmap.fromImage(qt_image))

    def on_rename_clicked(self):

        # Parameters: parent, title, label, text=default_value
        # Stores the new name given by the user and the bool "ok" becomes true if the user clicked ok and false if they clicked cancel or closed the window
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Face",
            "Enter new name:",

            # prefills the textbox with the current name
            text=self.display_name
        )

        # If the user did not click cancel and did not leave the box blank
        if ok and new_name and new_name != self.display_name:
            # Emits the signal defined in constructor
            self.rename_requested.emit(self.name, new_name)

            # Update label/UI to say new name
            self.name_label.setText(new_name)
            self.display_name = new_name

class FaceRecognitionApp(QMainWindow):
    # Main window of the application

    def __init__(self):
        super().__init__()

        # Loads the mediapipe models
        self.detector = FaceDetector()

        # Loads the index file
        self.database = FaceDatabase("data")

        # Tells the computer to open the webcam
        self.camera = cv2.VideoCapture(0)

        # Helps the app decide whether to show the live cama or an uploaded photo
        self.current_mode = None  # 'live', 'image', or None
        self.current_image = None

        # Stores where the faces are in the space of the current frame so that the boxes can be drawn
        self.current_face_locations = []
        self.current_face_landmarks = None

        # For UI you have to use the QTimer instead of a while loop to update the feed and grab a new frame from the camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)

        # Setup UI and load faces
        self.setup_ui()
        self.load_saved_faces()

    def setup_ui(self):
        """Initialize the user interface."""

        # TODO: Set window title
        self.setWindowTitle("AI Powered Face Recognition System")

        # TODO: Set window geometry (x, y, width, height)
        self.setGeometry(100, 100, 1200, 800)

        # TODO: Create and set central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # TODO: Create main horizontal layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # TODO: Create left vertical layout for display area
        left_layout = QVBoxLayout()

        # Create toolbar
        toolbar = QToolBar()

        # TODO: Create Live Feed button with emoji "📹"
        self.live_btn = QPushButton("📹 Live Feed")
        self.live_btn.clicked.connect(self.start_live_feed)
        toolbar.addWidget(self.live_btn)

        # TODO: Create Upload button with emoji "📁"
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(self.upload_btn)

        # TODO: Create Save Face button with emoji "💾"
        self.save_face_btn = QPushButton("💾 Save Face")
        self.save_face_btn.clicked.connect(self.save_current_face)
        self.save_face_btn.setEnabled(False)  # Disabled initially
        toolbar.addWidget(self.save_face_btn)

        left_layout.addWidget(toolbar)

        # TODO: Create display label for video/image
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.display_label.setText("Click 'Live Feed' or 'Upload Image' to start")
        left_layout.addWidget(self.display_label)

        # TODO: Add left_layout to main_layout with stretch factor 3
        main_layout.addLayout(left_layout, 3)

        # TODO: Create right vertical layout
        right_layout = QVBoxLayout()

        # Search label
        search_label = QLabel("Search Faces:")
        right_layout.addWidget(search_label)

        # TODO: Create search input field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search...")
        self.search_input.textChanged.connect(self.on_search_changed)
        right_layout.addWidget(self.search_input)

        # TODO: Create scroll area for faces list
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(300)

        # TODO: Create container widget for faces
        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setAlignment(Qt.AlignTop)
        self.faces_container.setLayout(self.faces_layout)

        self.scroll_area.setWidget(self.faces_container)
        right_layout.addWidget(self.scroll_area)

        # TODO: Add right_layout to main_layout with stretch factor 1
        main_layout.addLayout(right_layout, 1)

        # TODO: Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def start_live_feed(self):
        """Start or stop live camera feed."""

        # TODO: Check if currently in live mode
        if self.current_mode == 'live':
            # Stop camera
            self.timer.stop()
            self.camera.stop()
            self.current_mode = None
            self.live_btn.setText("📹 Live Feed")
            self.display_label.setText("Live feed stopped")
            self.save_face_btn.setEnabled(False)
            self.status_bar.showMessage("Live feed stopped")
            return

        # TODO: Try to start camera
        if self.camera.start():
            self.current_mode = 'live'

            # TODO: Start timer with 30ms interval (~30 FPS)
            self.timer.start(30)

            self.live_btn.setText("⏹️ Stop Feed")
            self.status_bar.showMessage("Live feed started")
        else:
            # TODO: Show warning message box
            QMessageBox.warning(self, "Camera Error", "Failed to start camera")

    def update_live_feed(self):
        """Update live feed frame (called by timer)."""

        # TODO: Read frame from camera
        frame = self.camera.read()
        if frame is None:
            return

        # TODO: Store copy as current_image
        self.current_image = frame.copy()

        # TODO: Detect faces and landmarks
        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(frame)

        # TODO: Draw face boxes if faces detected
        if self.current_face_locations:
            frame = self.detector.draw_faces(frame, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)

        # TODO: Draw landmarks if detected
        if self.current_face_landmarks:
            frame = self.detector.draw_landmarks(frame, self.current_face_landmarks)

        # TODO: Try to match first face
        if self.current_face_locations:
            first_face = self.current_face_locations[0]

            # TODO: Get face encoding
            encoding = self.detector.get_encoding(self.current_image, first_face)

            if encoding is not None:
                # TODO: Find closest match in database
                match = self.database.find_closest_match(encoding)

                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        # TODO: Display frame
        self.display_frame(frame)

    def upload_image(self):
        """Upload and process a static image."""

        # TODO: Open file dialog using QFileDialog.getOpenFileName()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return  # User cancelled

        # Stop live feed if running
        if self.current_mode == 'live':
            self.timer.stop()
            self.camera.stop()
            self.live_btn.setText("📹 Live Feed")

        self.current_mode = 'image'

        # TODO: Load image using cv2.imread()
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image")
            return

        # TODO: Store copy and detect faces
        self.current_image = image.copy()
        self.current_face_locations, self.current_face_landmarks = \
            self.detector.detect_faces(image)

        # TODO: Draw boxes if faces found
        if self.current_face_locations:
            image = self.detector.draw_faces(image, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
        else:
            self.save_face_btn.setEnabled(False)
            QMessageBox.information(self, "No Faces", "No faces detected in image")

        # TODO: Draw landmarks
        if self.current_face_landmarks:
            image = self.detector.draw_landmarks(image, self.current_face_landmarks)

        # TODO: Try to match first face
        if self.current_face_locations:
            first_face = self.current_face_locations[0]
            encoding = self.detector.get_encoding(self.current_image, first_face)

            if encoding is not None:
                match = self.database.find_closest_match(encoding)
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        # Display image
        self.display_frame(image)

    def save_current_face(self):
        """Save the first detected face to the database."""

        # TODO: Validate we have a face
        if not self.current_face_locations or self.current_image is None:
            QMessageBox.warning(self, "No Face", "No face detected to save")
            return

        # TODO: Prompt user for name using QInputDialog.getText()
        name, ok = QInputDialog.getText(self, "Save Face", "Enter person's name:")
        if not ok or not name:
            return

        # Get first face
        first_face = self.current_face_locations[0]

        # TODO: Crop face using detector
        face_image = self.detector.crop_face(self.current_image, first_face)

        # TODO: Generate encoding
        encoding = self.detector.get_encoding(self.current_image, first_face)
        if encoding is None:
            QMessageBox.warning(self, "Error", "Failed to generate face encoding")
            return

        # TODO: Save to database
        if self.database.save_face(name, face_image, encoding):
            QMessageBox.information(self, "Success", f"Face saved as '{name}'")
            self.load_saved_faces()  # Refresh list
        else:
            QMessageBox.warning(self, "Error", "Failed to save face")

    def load_saved_faces(self, search_query: str = ""):
        """Load and display saved faces in the sidebar."""

        # TODO: Clear existing widgets (iterate in reverse)
        for i in reversed(range(self.faces_layout.count())):
            widget = self.faces_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # TODO: Get faces from database (filtered or all)
        if search_query:
            faces = self.database.search_faces(search_query)
        else:
            faces = self.database.get_all_faces()

        # TODO: Create widget for each face
        for face in faces:
            widget = FaceItemWidget(
                face['name'],
                face['display_name'],
                face['face_path']
            )
            # TODO: Connect rename signal
            widget.rename_requested.connect(self.on_face_renamed)
            self.faces_layout.addWidget(widget)

    def on_search_changed(self, text: str):
        """Handle search input changes."""
        # TODO: Reload faces with filter
        self.load_saved_faces(text)

    def on_face_renamed(self, old_name: str, new_name: str):
        """Handle face rename request."""
        # TODO: Rename in database
        if self.database.rename_face(old_name, new_name):
            self.status_bar.showMessage(f"Renamed to '{new_name}'")

    def display_frame(self, frame: np.ndarray):
        """Display a frame in the GUI."""

        # TODO: Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate resize to fit display
        h, w, ch = rgb_frame.shape
        display_width = self.display_label.width()
        display_height = self.display_label.height()

        # Calculate aspect ratio
        aspect = w / h
        if display_width / display_height > aspect:
            new_height = display_height
            new_width = int(new_height * aspect)
        else:
            new_width = display_width
            new_height = int(new_width / aspect)

        # TODO: Resize frame using cv2.resize()
        resized = cv2.resize(rgb_frame, (new_width, new_height))
        h, w, ch = resized.shape

        # TODO: Convert to Qt image
        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line,
                           QImage.Format_RGB888)

        # TODO: Set pixmap on display label
        self.display_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        # TODO: Stop timer
        self.timer.stop()

        # TODO: Stop camera
        self.camera.stop()

        # TODO: Cleanup detector
        self.detector.cleanup()

        # TODO: Accept event to allow close
        event.accept()


def main():
    """Main entry point for the application."""

    # TODO: Create Qt application
    app = QApplication(sys.argv)

    # TODO: Create main window
    window = FaceRecognitionApp()

    # TODO: Show window
    window.show()

    # TODO: Start event loop and exit with return code
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
