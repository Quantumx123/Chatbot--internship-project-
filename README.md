# ⚡ Zeus Chatbot

An intelligent, AI-powered conversational agent for customer service built as a capstone project. Originally a basic CLI tool, Zeus has been upgraded into a production-ready, interactive web application featuring a modern, dark-themed, glassmorphic UI.

## 📋 Project Overview

Zeus demonstrates end-to-end development of an NLP-based customer service chatbot:

1. **Data Collection** — A structured intents dataset with 31 comprehensive customer service categories, ranging from order tracking to warranty information.
2. **Text Preprocessing** — Tokenization, lemmatization, stopword removal, and TF-IDF vectorization.
3. **Model Training** — Multi-class intent classification using scikit-learn (LinearSVC). The model achieved 99% accuracy on the full training dataset.
4. **Context Management & Entity Extraction** — The NLP engine uses state machines to extract context (like order numbers) and clarify vague inputs (e.g., asking users to clarify general "order issues").
5. **Deployment** — A production-ready Flask backend serving a responsive, dynamic HTML/JS/CSS web interface.

## 🏗️ Project Structure

```
Chatbot (internship project)/
├── app.py                     # Flask web server API & UI router
├── templates/
│   └── index.html             # Main chatbot web interface
├── static/
│   ├── style.css              # Glassmorphic, dark-theme styling
│   └── script.js              # Frontend chat logic (auto-scroll, typing animation)
├── data/
│   └── intents.json           # Training dataset (31 intents)
├── nlp/
│   ├── __init__.py
│   └── preprocessing.py       # Text cleaning & NLP pipeline
├── model/
│   ├── __init__.py
│   ├── train.py               # Model training & evaluation
│   ├── trained_model.joblib   # Saved model weights
│   └── vectorizer.joblib      # Saved TF-IDF vectorizer
├── chatbot/
│   ├── __init__.py
│   └── chatbot.py             # Core NLP Engine, State Machine & CLI interface
├── requirements.txt
└── README.md
```

## 🚀 Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model/train.py
```

This will preprocess the dataset, train the `LinearSVC` model via 5-fold cross-validation, and generate `.joblib` files inside the `model/` directory.

### 3. Run the Chatbot

**Web Interface (Recommended):**
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000` to interact with Zeus.

**CLI Interface:**
```bash
python chatbot/chatbot.py
```

## 💬 Supported Capabilities (31 Intents)

Zeus is trained on a wide variety of customer support interactions:

- **General:** Greetings, Goodbyes, Thanks, Acknowledgements.
- **Order Management:** Check order status, Track a package, Cancel an order, Change an order.
- **Problem Resolution:** Refund requests, Return policy, Damaged items, Missing packages, General order complaints.
- **Store Information:** Pricing, Product info, Promo codes, Payment methods, Warranty info, Store locations.
- **Customer Account:** Subscription help, Size guides, Out of stock notifications.
- **Support:** Hours of operation, Contact info, Escalate to human agent.
- **Session Logic:** Closing the chat session, Ambiguity resolution.

## 🧠 Advanced NLP Features

- **Entity Extraction:** Zeus uses Regex to pull 5-to-8 digit order numbers dynamically from user inputs.
- **State Machine / Context Memory:** If Zeus needs an order number (e.g., to cancel an order), it enters an `awaiting_order_number` state and waits for the user to provide it.
- **Ambiguity Resolution:** Vague statements like *"issue with my order"* trigger Zeus to ask clarifying questions before taking action.
- **Confidence Thresholding:** Queries falling below a 25% confidence threshold trigger a fallback "I don't understand" response.

## 🛠️ Technology Stack

- **Backend:** Python 3.10+, Flask
- **Frontend:** Vanilla HTML5, CSS3, JavaScript
- **NLP / ML:** NLTK, scikit-learn (TF-IDF, LinearSVC), joblib
