# 🤖 Customer Service Chatbot

An intelligent, text-based conversational agent for customer service built as a capstone project. The chatbot uses Natural Language Processing (NLP) and Machine Learning to understand user intent and provide helpful responses.

## 📋 Project Overview

This project demonstrates end-to-end development of a customer service chatbot:

1. **Data Collection** — A structured intents dataset with 18 customer service categories
2. **Text Preprocessing** — Tokenization, lemmatization, stopword removal, and TF-IDF vectorization
3. **Model Training** — Multi-class intent classification using scikit-learn (LinearSVC, Logistic Regression, Multinomial Naive Bayes)
4. **Deployment & Testing** — Interactive CLI chatbot with confidence scoring

## 🏗️ Project Structure

```
Chatbot (internship project)/
├── data/
│   └── intents.json           # Training dataset (18 intents)
├── nlp/
│   ├── __init__.py
│   └── preprocessing.py       # Text cleaning & NLP pipeline
├── model/
│   ├── __init__.py
│   ├── train.py               # Model training & evaluation
│   ├── trained_model.joblib   # Saved model (generated)
│   └── vectorizer.joblib      # Saved vectorizer (generated)
├── chatbot/
│   ├── __init__.py
│   └── chatbot.py             # Chatbot engine & CLI interface
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

This will:
- Load the intents dataset
- Preprocess all training patterns
- Compare 3 classifiers via 5-fold cross-validation
- Train the best model on all data
- Save the model and vectorizer to `model/`

### 3. Run the chatbot

```bash
python chatbot/chatbot.py
```

## 💬 Supported Intents

| Intent | Description | Example Query |
|--------|-------------|---------------|
| `greeting` | Hello, hi, hey | "Hi there!" |
| `goodbye` | Farewell messages | "Bye, see you later" |
| `thanks` | Gratitude | "Thank you so much" |
| `order_status` | Check order status | "Where is my order?" |
| `track_order` | Track a package | "Can I track my delivery?" |
| `cancel_order` | Cancel an order | "I want to cancel my order" |
| `return_policy` | Return information | "What is your return policy?" |
| `refund_request` | Request a refund | "I want my money back" |
| `product_info` | Product details | "Tell me about your products" |
| `pricing` | Price inquiries | "How much does it cost?" |
| `shipping_info` | Shipping options | "Do you offer free shipping?" |
| `delivery_time` | Delivery estimates | "How long does delivery take?" |
| `payment_issues` | Payment problems | "My payment failed" |
| `account_help` | Account assistance | "I forgot my password" |
| `complaint` | File a complaint | "I'm not satisfied" |
| `escalate_to_human` | Talk to a human | "Let me speak to a manager" |
| `hours_of_operation` | Business hours | "When are you open?" |
| `contact_info` | Contact details | "What is your phone number?" |

## 🛠️ Technology Stack

- **Python 3.10+**
- **NLTK** — Tokenization, stopword removal, lemmatization
- **scikit-learn** — TF-IDF vectorization, model training (LinearSVC, LogisticRegression, MultinomialNB)
- **joblib** — Model serialization

## 📊 Model Details

- **Vectorizer:** TF-IDF with unigram + bigram features
- **Best Classifier:** Selected automatically via 5-fold stratified cross-validation
- **Confidence Threshold:** 25% — queries below this threshold trigger a fallback response
