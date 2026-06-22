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
import re

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
        self.session_state = None
        self.current_intent = None

    # ----- entity extraction -----------------------------------------------

    def extract_order_number(self, text: str) -> str | None:
        """Extract a 5 to 8 digit order number from text."""
        # Matches numbers with 5 to 8 digits, optionally prefixed by order/#/ord
        match = re.search(r'\b\d{5,8}\b', text)
        if match:
            return match.group(0)
        return None

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
        # 1. Check if we are waiting for an order number
        if self.session_state == "awaiting_order_number":
            order_number = self.extract_order_number(user_input)
            if order_number:
                # Handle context based on what intent asked for it
                if self.current_intent == "cancel_order":
                    response = f"Alright, I have successfully submitted a cancellation request for order {order_number}."
                elif self.current_intent == "order_status":
                    response = f"Checking order {order_number}... It looks like it is currently processing and will ship soon!"
                elif self.current_intent == "track_order":
                    response = f"Tracking order {order_number}... Your package is out for delivery today!"
                elif self.current_intent == "refund_request":
                    response = f"I've initiated the refund process for order {order_number}. It should appear in 5-7 business days."
                elif self.current_intent == "change_order":
                    response = f"I pulled up order {order_number}. I will transfer you to an agent who can modify it for you."
                elif self.current_intent == "damaged_item":
                    response = f"I've located order {order_number}. Let's get a replacement or refund sorted out right away."
                else:
                    response = f"Thanks for the order number ({order_number}). I've updated your account."
                
                # Reset state
                self.session_state = None
                self.current_intent = None
                
                return {
                    "response": response,
                    "intent": "context_resolved",
                    "confidence": 1.0,
                }
            else:
                return {
                    "response": "I didn't quite catch an order number there. It should be a 5-to-8 digit number. Could you try again?",
                    "intent": "awaiting_order_number",
                    "confidence": 1.0,
                }

        # 1.5. Check if we are waiting for a close decision
        if self.session_state == "awaiting_close_decision":
            text_lower = user_input.lower()
            if any(word in text_lower for word in ["close", "end", "yes", "that's all", "nothing", "no"]):
                self.session_state = None
                self.current_intent = None
                return {
                    "response": "Thank you for reaching out! The chat session is now closed. Have a great day!",
                    "intent": "close_session",
                    "confidence": 1.0,
                    "action": "close_session"
                }
            else:
                self.session_state = None
                self.current_intent = None
                # If they say something else, treat it as a new question
                # Just drop down to normal intent prediction

        # 2. Extract entities immediately in case user provided them upfront
        order_number = self.extract_order_number(user_input)

        # 3. Predict Intent
        intent, confidence = self.predict_intent(user_input)

        # Fall back if confidence is too low
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "fallback"

        # 4. Handle early entity provision (skip awaiting state)
        order_intents = ["cancel_order", "order_status", "track_order", "refund_request", "change_order", "damaged_item"]
        if intent in order_intents and order_number:
            if intent == "cancel_order":
                response = f"Alright, I have successfully submitted a cancellation request for order {order_number}."
            elif intent == "order_status":
                response = f"Checking order {order_number}... It looks like it is currently processing and will ship soon!"
            elif intent == "track_order":
                response = f"Tracking order {order_number}... Your package is out for delivery today!"
            elif intent == "refund_request":
                response = f"I've initiated the refund process for order {order_number}. It should appear in 5-7 business days."
            elif intent == "change_order":
                response = f"I pulled up order {order_number}. I will transfer you to an agent who can modify it for you."
            elif intent == "damaged_item":
                response = f"I've located order {order_number}. Let's get a replacement or refund sorted out right away."
                
            self.session_state = None
            self.current_intent = None
            return {
                "response": response,
                "intent": intent,
                "confidence": round(confidence, 4),
            }

        # 5. Select a random response for the intent
        if intent == "greeting":
            if self.greeted:
                response = "How else can I help you?"
            else:
                self.greeted = True
                response = random.choice(self.responses.get(intent, ["Hello!"]))
        elif intent == "close_session":
            response = random.choice(self.responses.get(intent, ["Session closed."]))
            return {
                "response": response,
                "intent": intent,
                "confidence": round(confidence, 4),
                "action": "close_session"
            }
        else:
            response = random.choice(
                self.responses.get(intent, ["I'm not sure how to help with that."])
            )

        # 6. Check if response sets a context state
        if "[CONTEXT:awaiting_order_number]" in response:
            response = response.replace("[CONTEXT:awaiting_order_number]", "").strip()
            self.session_state = "awaiting_order_number"
            self.current_intent = intent
        elif "[CONTEXT:awaiting_close_decision]" in response:
            response = response.replace("[CONTEXT:awaiting_close_decision]", "").strip()
            self.session_state = "awaiting_close_decision"
            self.current_intent = intent

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
    print("|                          ZEUS                           |")
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

    print("Bot  >>  Hello, I am Zeus, How may I assist you today?")
    print()

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
        
        if result.get("action") == "close_session":
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_cli()
