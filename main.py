import cv2
import os
import json
import pandas as pd
from datetime import datetime
from modules.vision_module import VisionModule
from modules.speech_module import SpeechModule
from modules.nlp_module import NLPModule
import speech_recognition as sr

# Initialize NLP module
nlp = NLPModule()

# Train the NLP model if it doesn't exist
if not os.path.exists(nlp.model_path):
    print("Training NLP model from JSON data...")
    nlp.train()

# Initialize vision and speech modules
vision = VisionModule()
speech = SpeechModule()

logs = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live Smart Classroom Assistant. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Smart Classroom Assistant", frame)

    attendance = vision.detect_face_and_mood(frame)
    print(f"Detected {len(attendance)} faces")

    for i, (name, status, mood) in enumerate(attendance):
        print(f"[{i}] Detected: {name} Mood: {mood}")
        print("Listening for question...")

        question = None
        for attempt in range(3):
            try:
                with sr.Microphone() as source:
                    speech.recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = speech.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    question = speech.recognizer.recognize_google(audio)
                if question:
                    break
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start")
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Google Speech Recognition service error: {e}")
                break
            print(f"Retrying listening... attempt {attempt+1}")

        if question:
            print(f"Question: {question}")
            topic = nlp.predict_topic(question)
            print(f"Topic: {topic}")
        else:
            topic = None
            print("No valid question received.")

        log_entry = {
            "student": name,
            "attendance": status,
            "mood": mood,
            "question": question,
            "topic": topic,
            "timestamp": str(datetime.now())
        }
        logs.append(log_entry)
        print(f"Log entry added: {log_entry}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

os.makedirs("data/logs", exist_ok=True)
with open("data/logs/logs.json", "w") as f:
    json.dump(logs, f)

pd.DataFrame(logs).to_csv("data/logs/logs.csv", index=False)

print(f"Logs saved with {len(logs)} entries.")

cap.release()
cv2.destroyAllWindows()
