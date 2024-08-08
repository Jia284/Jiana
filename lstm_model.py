from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import langdetect
from flask import jsonify
import traceback
import pickle

model_path = 'C:/Users/JIANA/Downloads/project practice with ai ml/project practice with ai ml/html/spam_det.keras'
model = load_model(model_path)
with open('lstm_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 50  

def detect_spam_lstm(text):
    try:
        lang = langdetect.detect(text)
        if lang != 'en':
            return {'message': "Language not supported.", "justification":""}

        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        prediction = model.predict(padded_sequence)

        if prediction > 0.5:
            message = "This is a spam."
        else:
            message = "This is not a spam."

        return {'message': message, "justification":""}
    except Exception as e:
        #print(e)
        print("LSTM Exception: ", traceback.format_exc())
        return {'message': "API error"}
