
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import langdetect
from flask import jsonify
import traceback

with open('random_new.pkl', 'rb') as random_file:
    random_vectorizer = pickle.load(random_file)

with open('model_new_rf.pkl', 'rb') as tree_file:
    random_classifier = pickle.load(tree_file)

def detect_spam_random_forest(text):
    try:
        lang = langdetect.detect(text)
        if lang != 'en':
            return {'message': "Language not supported.", "justification":""}

        text_transformed = random_vectorizer.transform([text])
        prediction = random_classifier.predict(text_transformed)

        if prediction == 1:
            message = "This is a spam."
        else:
            message = "This is not a spam."

        return {'message': message, "justification":""}
    except:
        print("Random Forest Exception: ", traceback.format_exc())
        return {'message': f"API error {traceback.format_exc()}", "justification":""}