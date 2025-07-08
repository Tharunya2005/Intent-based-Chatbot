from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import json
import random
import nltk
import ssl
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Fix SSL issues (important for some systems)
try:
    _create_unverified_https_context = ssl._create_default_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- NLTK Data Setup (Updated for Render deployment) ---
# Define the directory where NLTK data will be stored on Render
# This should match the download_dir in your render.yaml buildCommand
nltk_data_dir = "./.nltk_data" 

# Ensure the directory exists (might be created by buildCommand, but good practice)
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the custom NLTK data path to NLTK's search path
nltk.data.path.append(nltk_data_dir)

# Download NLTK data if not already present (primarily for local development,
# as Render's buildCommand will handle it during deployment)
# We moved the primary download to render.yaml's build command for deployment reliability.
# This block is mainly for local testing and initial setup.
for dataset in ["punkt", "wordnet", "omw-1.4"]: # Ensure all needed datasets are listed
    try:
        # For 'punkt', it's under 'tokenizers/punkt'
        # For 'wordnet' and 'omw-1.4', they are directly at the dataset root
        if dataset == "punkt":
            nltk.data.find(f"tokenizers/{dataset}")
        else:
            nltk.data.find(dataset)
    except LookupError: # Use LookupError when checking if resource is found
        print(f"Downloading NLTK dataset: {dataset}...")
        nltk.download(dataset, download_dir=nltk_data_dir)
        print(f"Finished downloading {dataset}.")

# NLP Setup
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in words]

# Load intents JSON
file_path = os.path.abspath("Intent.json")
try:
    with open(file_path, "r", encoding="utf-8") as file:
        intents = json.load(file)
except Exception as e:
    print(f"❌ Error loading intents: {e}")
    intents = {"intents": []}

# Core chatbot logic
def chatbot(input_text):
    input_words = preprocess_text(input_text)
    best_match = None
    max_overlap = 0

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = preprocess_text(pattern)
            overlap = len(set(input_words) & set(pattern_words))
            if overlap > max_overlap:
                best_match = intent
                max_overlap = overlap

    if best_match:
        return random.choice(best_match["responses"])
    return "I'm still learning, please rephrase your question."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    history_data = []
    try:
        with open('chat_log.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header if exists
            history_data = list(reader)[-10:]  # Show last 10 entries
    except FileNotFoundError:
        history_data = []

    return render_template("history.html", history=history_data)

@app.route("/get-response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response = chatbot(user_input)

    # Save conversation to chat log
    log_exists = os.path.exists('chat_log.csv')
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["User Input", "Chatbot Response"])
        writer.writerow([user_input, response])

    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))