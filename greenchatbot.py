import json
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def load_intents(filename):
  """
  Loads intents from a JSON file.

  Args:
    filename: The name of the JSON file containing the intents.

  Returns:
    A dictionary containing the loaded intents.
  """
  with open(filename, 'r') as f:
    return json.load(f)

def preprocess_text(text):
  """
  Preprocesses the input text by converting to lowercase and lemmatizing words.

  Args:
    text: The input text string.

  Returns:
    A list of preprocessed words.
  """
  words = word_tokenize(text.lower())
  words = [lemmatizer.lemmatize(word) for word in words]
  return words

def get_response(user_input, intents):
  """
  Determines the intent of the user's input and provides an appropriate response.

  Args:
    user_input: The user's input as a string.
    intents: A dictionary containing the loaded intents.

  Returns:
    A string containing the chatbot's response.
  """
  user_input_words = preprocess_text(user_input)

  # Find potential matches based on word overlap
  potential_matches = []
  for intent in intents["intents"]:
    for pattern in intent["patterns"]:
      pattern_words = preprocess_text(pattern)
      common_words = set(user_input_words) & set(pattern_words)
      if len(common_words) > 0:
        potential_matches.append((intent, len(common_words)))

  # Select the best match (intent with the most overlapping words)
  if potential_matches:
    best_match = max(potential_matches, key=lambda x: x[1])[0]
    return random.choice(best_match["responses"])

  # If no good match is found, return a "still learning" response
  return random.choice([
      "I'm still learning, can you please rephrase your question?",
      "I'm still under development, but I'm learning new things every day!",
      "I'm not sure I understand, could you try asking that again?",
      "I'm still learning how to understand you better."
  ])

if __name__ == "__main__":
  intents = load_intents("Intent.json")  # Load intents before using them
  while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
      break
    response = get_response(user_input, intents)
    print("Chatbot:", response)