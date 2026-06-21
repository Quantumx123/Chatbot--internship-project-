"""
Chatbot Engine – Intent Classification & Response Selection.

This module loads the trained model and vectorizer, accepts user input,
predicts the intent, and returns an appropriate response. It also provides
a terminal-based interactive chat loop.
"""

import json
import os
import sys
import random

import joblib
import numpy as np

# Add project root to path so we can import the nlp package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlp.preprocessing import TextPreprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "intents.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "trained_model.joblib")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "model", "vectorizer.joblib")

# Confidence threshold – if the best prediction probability is below this
# value, the chatbot will use the fallback intent.
CONFIDENCE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Chatbot class
# ---------------------------------------------------------------------------
class CustomerServiceBot:
    """Intelligent customer-service chatbot powered by a trained classifier."""

    def __init__(self):
        # Load artefacts
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.preprocessor = TextPreprocessor()

        # Load intents for response lookup
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_data = json.load(f)

        # Build a tag -> responses mapping
        self.responses: dict[str, list[str]] = {}
        for intent in intents_data["intents"]:
            self.responses[intent["tag"]] = intent["responses"]

        # Session state
        self.greeted = False

    # ----- prediction ------------------------------------------------------

    def predict_intent(self, user_input: str) -> tuple[str, float]:
        """
        Predict the intent for the given user input.

        Returns:
            (intent_tag, confidence)
        """
        processed = self.preprocessor.preprocess(user_input)
        X = self.vectorizer.transform([processed])

        # Get probability estimates for all classes
        probas = self.model.predict_proba(X)[0]
        best_idx = np.argmax(probas)
        confidence = probas[best_idx]
        intent = self.model.classes_[best_idx]

        return intent, float(confidence)

    # ----- response selection ----------------------------------------------

    def get_response(self, user_input: str) -> dict:
        """
        Process user input and return a response dict with:
          - response (str)
          - intent  (str)
          - confidence (float)
        """
        intent, confidence = self.predict_intent(user_input)

        # Fall back if confidence is too low
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "fallback"

        # Avoid repeating the greeting
        if intent == "greeting":
            if self.greeted:
                response = "How else can I help you?"
            else:
                self.greeted = True
                response = random.choice(self.responses.get(intent, ["Hello!"]))
        else:
            response = random.choice(
                self.responses.get(intent, ["I'm not sure how to help with that."])
            )

        return {
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 4),
        }


# ---------------------------------------------------------------------------
# Interactive CLI chat loop
# ---------------------------------------------------------------------------
def run_cli():
    """Launch a terminal-based chat session."""

    # ---- Header banner ----
    print()
    print("+=========================================================+")
    print("|          CUSTOMER SERVICE CHATBOT                        |")
    print("|                                                          |")
    print("|   Type your question and press Enter.                    |")
    print("|   Type 'quit' or 'exit' to end the session.             |")
    print("+=========================================================+")
    print()

    try:
        bot = CustomerServiceBot()
    except FileNotFoundError:
        print("ERROR: Trained model not found!")
        print("Please run  'python model/train.py'  first to train the model.")
        sys.exit(1)

    while True:
        try:
            user_input = input("You  >>  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\nBot  >>  Goodbye! Have a wonderful day!")
            break

        result = bot.get_response(user_input)

        print(f"Bot  >>  {result['response']}")
        print(f"         [intent: {result['intent']}  |  confidence: {result['confidence']:.2%}]")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_cli()
