"""
Computer Vision Module for Smart Classroom Assistant

- Loads student face images (e.g., 'ahmed.jpg', 'ali.jpg')
- Uses filename (without '.jpg') as student name 
- Performs face recognition and mood detection on webcam frames
"""
# import libraries:
import face_recognition         # For face detection and recognition
from fer import FER            # For emotion detection
import os                      # For file system operations


class VisionModule:
    def __init__(self, students_dir="data/students"):
        self.students_dir = students_dir
        self.known_faces = []      # Stores face encodings
        self.known_names = []      # Stores student names (e.g., 'ahmed', 'ali')
        self.load_known_faces()    # Load data on module initialization

    def load_known_faces(self):
        # Loop through each image file in the students directory
        for img_file in os.listdir(self.students_dir):
            img_path = os.path.join(self.students_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            # Skip images if no face is detected
            if len(encodings) == 0:
                continue
            encoding = encodings[0]
            # Use filename (excluding extension) as the student name
            name = os.path.splitext(img_file)[0]
            self.known_faces.append(encoding)
            self.known_names.append(name)

    def detect_face_and_mood(self, frame):
        # Detect faces in the provided webcam frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
            names.append(name)
        # Run FER for emotion detection in the same frame
        detector = FER(mtcnn=True)
        detected_emotions = detector.detect_emotions(frame)
        moods = [max(e['emotions'], key=e['emotions'].get) if e else "unknown" for e in detected_emotions]
        # Return tuple of (student_name, attendance_status, detected_mood) for each detected face
        return list(zip(names, ['Present'] * len(names), moods))
