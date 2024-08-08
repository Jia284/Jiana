import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import langdetect
from flask import jsonify
import traceback


with open('vectorizer_new.pkl', 'rb') as models_file:
    vectorizer = pickle.load(models_file)

with open('model_new.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

def detect_spam_naive_bayes(text):
    try:
        lang = langdetect.detect(text)
        if lang != 'en':
            return jsonify({'message': "Language not supported.", "justification":""})

        text_transformed = vectorizer.transform([text])
        prediction = classifier.predict(text_transformed)

        if prediction == 1:
            message = "This is a spam."
        else:
            message = "This is not a spam."

        return {'message': message, "justification":""}
    except:
        print("Naive Bayes Exception: ", traceback.format_exc())
        return {'message': f"API error {traceback.format_exc()}", "justification":""}

