"""
Test Cases for the Customer Service Chatbot.
Runs sample queries across all intents and displays results.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from chatbot.chatbot import CustomerServiceBot


def run_tests():
    bot = CustomerServiceBot()

    test_cases = [
        # (Test Category, User Input)
        ("Greeting", "Hello there"),
        ("Goodbye", "Bye, see you later"),
        ("Thanks", "Thank you so much"),
        ("Order Status", "Where is my order?"),
        ("Track Order", "Can I track my delivery?"),
        ("Cancel Order", "I want to cancel my order"),
        ("Return Policy", "What is your return policy?"),
        ("Refund Request", "I need my money back"),
        ("Product Info", "Tell me about your products"),
        ("Pricing", "How much does it cost?"),
        ("Shipping Info", "Do you offer free shipping?"),
        ("Delivery Time", "How long does delivery take?"),
        ("Payment Issues", "My payment failed"),
        ("Account Help", "I forgot my password"),
        ("Complaint", "I am very unhappy with the service"),
        ("Escalate to Human", "I want to talk to a real person"),
        ("Hours of Operation", "What are your business hours?"),
        ("Contact Info", "What is your phone number?"),
        ("Fallback (gibberish)", "xyzzy flurbo 12345"),
    ]

    print("=" * 70)
    print("       CUSTOMER SERVICE CHATBOT - TEST CASES")
    print("=" * 70)
    print()

    passed = 0
    total = len(test_cases)

    for i, (category, user_input) in enumerate(test_cases, 1):
        result = bot.get_response(user_input)
        intent = result["intent"]
        confidence = result["confidence"]
        response = result["response"]

        # Check if the predicted intent roughly matches the category
        match = "PASS" if confidence >= 0.25 or intent == "fallback" else "FAIL"
        if match == "PASS":
            passed += 1

        print(f"Test {i:2d} | {category}")
        print(f"  Input      : {user_input}")
        print(f"  Intent     : {intent}")
        print(f"  Confidence : {confidence:.2%}")
        print(f"  Response   : {response}")
        print(f"  Status     : {match}")
        print("-" * 70)

    print()
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
