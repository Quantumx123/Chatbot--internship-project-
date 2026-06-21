"""
Flask Web Application for the Customer Service Chatbot.

Routes:
  GET  /     — Serve the chat UI
  POST /chat — Accept user message, return bot response as JSON
"""

import os
import sys

from flask import Flask, render_template, request, jsonify

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from chatbot.chatbot import CustomerServiceBot  # noqa: E402

app = Flask(__name__)

# Initialize chatbot once at startup
bot = CustomerServiceBot()


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message and return the bot's response."""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = bot.get_response(user_message)
    return jsonify(result)


if __name__ == "__main__":
    print("\n  Customer Service Chatbot - Web Interface")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
