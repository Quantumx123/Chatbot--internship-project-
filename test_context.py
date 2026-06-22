"""
Test script for Multi-turn Conversation Context & Entity Extraction
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from chatbot.chatbot import CustomerServiceBot

def run_context_test():
    bot = CustomerServiceBot()
    
    print("=" * 60)
    print("  TEST 1: Multi-turn Context (Asking for Order Number)")
    print("=" * 60)
    
    # Turn 1: User asks to cancel order (Intent: cancel_order)
    user_input_1 = "I want to cancel my order"
    print(f"User: {user_input_1}")
    res_1 = bot.get_response(user_input_1)
    print(f"Bot : {res_1['response']}")
    print(f"[State after Turn 1: {bot.session_state}]")
    print()
    
    # Turn 2: User gives a raw order number (Normally a fallback!)
    user_input_2 = "88392"
    print(f"User: {user_input_2}")
    res_2 = bot.get_response(user_input_2)
    print(f"Bot : {res_2['response']}")
    print(f"[State after Turn 2: {bot.session_state}]")
    print("\n" + "=" * 60 + "\n")
    
    print("=" * 60)
    print("  TEST 2: Early Entity Extraction (Providing Number Upfront)")
    print("=" * 60)
    
    # Turn 1: User asks to track order WITH the number included
    user_input_3 = "Where is my package? The tracking is ORD-99281"
    print(f"User: {user_input_3}")
    res_3 = bot.get_response(user_input_3)
    print(f"Bot : {res_3['response']}")
    print(f"[State after Turn 1: {bot.session_state}]")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_context_test()
