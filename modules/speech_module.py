"""
Speech Module

This module uses the SpeechRecognition library to convert spoken
questions from a microphone into text using Google's speech-to-text API.
"""

import speech_recognition as sr

class SpeechModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()  # Initialize the recognizer

    def listen_and_transcribe(self):
        """Listens to the microphone and transcribes speech to text.
        Returns:
            str: The transcribed text, or None if recognition fails.
        """
        with sr.Microphone() as source:  # Use default microphone
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                # Use Google Web Speech API to convert speech to text
                text = self.recognizer.recognize_google(audio)
                return text
            except Exception as e:
                print(f"Error: {e}") # If any error occurs print the error message
                return None  # Return None to indicate failure to recognize speech
