"""
NLP Module for Smart Classroom Assistant

Uses training data from a JSON file to train the TF-IDF vectorizer and classifier.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
import json
import re


def preprocess_texts(texts):
    return [re.sub(r"[^\w\s]", "", text.lower()) for text in texts]


class NLPModule:
    def __init__(
        self,
        model_path="models/classifier.pkl",
        training_data_path="models/training_data.json",
    ):
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.topics = ["Math", "Computer Vision", "Programming", "Other"]

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.vectorizer, self.classifier = pickle.load(f)
        else:
            print("No trained model found. Please train the model before prediction.")

    def train(self):
        if not os.path.exists(self.training_data_path):
            raise FileNotFoundError(
                f"Training data file not found at {self.training_data_path}"
            )
        with open(self.training_data_path, "r") as f:
            data = json.load(f)
            X = data["questions"]
            y = data["topics"]

        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vectorizer, self.classifier), f)
        print(f"Model trained and saved to {self.model_path}")

    def predict_topic(self, text):
        X_vec = self.vectorizer.transform([text])
        pred = self.classifier.predict(X_vec)[0]
        if pred in self.topics:
            return pred
        else:
            return "Other"
