# Smart Classroom Assistant
The **Smart Classroom Assistant** integrates Computer Vision, Speech Recognition, and Natural Language Processing (NLP) to automate attendance, detect student moods, and classify questions by topic during classroom sessions.

## 🚀 Features:
- Automatically records attendance by recognizing student faces.
- Detects student moods (e.g., happy, tired) using emotion recognition.
- Converts spoken student questions to text with speech recognition.
- Classifies transcribed questions into topics such as Math, Programming, or Computer Vision.
- Logs attendance, mood, questions, and topics with timestamps.

## 🗂 Project Structure
```
smart-classroom-assistant/
│
├── modules/
│ ├── vision_module.py # Face recognition & mood detection
│ ├── speech_module.py # Speech-to-text transcription
│ ├── nlp_module.py # Question topic classification
│
├── models/
│ ├── classifier.pkl # Trained NLP model
│ ├── training_data.json # Training data for NLP module
│
├── data/
│ ├── students/ # Student face images (e.g., ahmed.jpg)
│ ├── logs/
│  ├── logs.json # Recorded log data
│  ├── logs.csv # Log data in CSV format
│
├── main.py # Main application script
├── requirements.txt # Required Python packages
└── README.md # Project documentation
```

## ▶️ Usage
1. Add student face images to `data/students/` with filenames as the student names (e.g., `ahmed.jpg`).
2. Install dependencies:
```pip install -r requirements.txt```
3. Run the assistant:
4. The system will:
- Open the webcam to detect and recognize students.
- Detect student moods.
- Listen to student questions via microphone.
- Transcribe and classify questions by topic.
- Save logs in `data/logs/` as JSON and CSV files.

## 📦 Requirements
- Python 3.8+
- opencv-python
- face_recognition
- fer (Facial Emotion Recognition)
- SpeechRecognition
- scikit-learn
- pandas
- torch
- moviepy==1.0.3
- tensorflow==2.11

---

Created with Salma Essam
