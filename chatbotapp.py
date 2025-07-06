import os
import json
import random
import nltk
import ssl
import streamlit as st
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Handle SSL certificate issues (important for deployment)
try:
    _create_unverified_https_context = ssl._create_default_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK data is downloaded (punkt_tab and wordnet for deployment)
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK datasets if missing
for dataset in ["punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{dataset}" if dataset == "punkt_tab" else dataset)
    except LookupError:
        nltk.download(dataset, download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Load intents from the JSON file
file_path = os.path.abspath("./Intent.json")  # Ensure Intent.json is in the same directory
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()
except json.JSONDecodeError:
    st.error(f"Invalid JSON format in file: {file_path}")
    st.stop()

# Initialize lemmatizer for preprocessing
lemmatizer = WordNetLemmatizer()

# Preprocess text by lemmatizing
def preprocess_text(text):
    words = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in words]

# Function to get chatbot response
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

counter = 0

# Adding custom HTML and CSS for the sustainable food practices theme
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background-color: #E4F1DC;
        font-family: 'Arial', sans-serif;
        color: #3A6A40;
        line-height: 1.6;
    }

    h1, h2 {
        color: #2A7B35;
    }

    .stButton>button {
        background-color: #A3C36B;
        color: white;
        border-radius: 8px;
        border: 2px solid #A3C36B;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #7C9F47;
        border-color: #7C9F47;
    }

    .stTextArea {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #A3C36B;
    }

    .stTextInput>div>input {
        background-color: #F1F9E7;
        color: #3A6A40;
        border: 1px solid #A3C36B;
        border-radius: 8px;
        padding: 8px 12px;
    }

    .stTextInput>div>input:focus {
        outline-color: #7C9F47;
    }

    .stMarkdown {
        margin-top: 20px;
    }

    .history-item {
        background-color: #F9F9F9;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #A3C36B;
    }

    .history-item-user {
        background-color: #86A788;
    }

    .history-item-chatbot {
        background-color: #638C6D;
        color: white;
    }

    .biofeast-header {
        text-align: center;
        background-color: #A9BFA8;
        padding: 20px;
        color: Olivegreen;
        font-size: 2.5em;
        font-weight: bold;
    }

    /* Sidebar Menu Customization */
    .sidebar .sidebar-content {
        background-color: #D39D55; /* Brown Soil color */
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    global counter
    add_custom_css()  # Apply custom CSS for the sustainable theme

    # Display BIOFEAST Header without logo
    st.markdown("""
    <div class="biofeast-header">
        BIOFEAST Chatbot
    </div>
    """, unsafe_allow_html=True)

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the Sustainable Food Practices Chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response'])

        counter += 1
        
        # Display conversation history at the top
        st.header("Conversation History:")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                history = list(csv_reader)[-5:]  # Show the last 5 conversations
                for row in history:
                    st.markdown(f"""
                    <div class="history-item history-item-user">User: {row[0]}</div>
                    <div class="history-item history-item-chatbot">Chatbot: {row[1]}</div>
                    """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.write("No conversation history yet.")

        # User input
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            # Get chatbot response
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history available.")

    # About Menu
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on sustainable food practices.")

        st.subheader("Project Overview:")
        st.write("""
        This chatbot is designed to help users understand and make informed decisions about sustainable food choices, reducing food waste, and supporting a healthier environment. The chatbot is trained using intents and machine learning techniques.
        """)

        st.subheader("How the Chatbot Works:")
        st.write("""
        The chatbot uses Natural Language Processing (NLP) techniques to process user input and match it to predefined intents. It then responds with relevant information related to sustainable food practices based on the user's query.
        """)

        st.subheader("Sustainable Food Practices:")
        st.write("""
        Sustainable food practices are methods of food production and consumption that have minimal environmental impact. This includes choosing locally sourced food, reducing food waste, supporting ethical farming practices, and eating more plant-based foods.
        """)

if __name__ == '__main__':
    main()