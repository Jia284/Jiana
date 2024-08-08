import pandas as pd
import langdetect
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import pickle

def detect_spam():
    new_data = pd.read_csv("spam_ham_dataset.csv")
    new_data.drop(columns=['Unnamed: 0'], inplace=True)

    def get_lang(text):
        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    new_data["lang"] = new_data["text"].apply(get_lang)
    pos = new_data[new_data["lang"]!="en"].index
    only_en = new_data.drop(index=pos)

    x_train, x_test, y_train, y_test = train_test_split(only_en["text"], only_en["label_num"], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(x_train)
    features_test_transformed = vectorizer.transform(x_test)

    with open('vectorizer_new.pkl', 'wb') as models_file:
        pickle.dump(vectorizer, models_file)

    classifier = MultinomialNB()
    classifier.fit(features_train_transformed, y_train)
    
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
    grid_search.fit(features_train_transformed, y_train)
    best_classifier = grid_search.best_estimator_


    accuracy = classifier.score(features_test_transformed, y_test)
    print("Classifier accuracy on new dataset: {:.2f}%".format(accuracy * 100))

    labels = classifier.predict(features_test_transformed)
    results = confusion_matrix(y_test, labels)
    print('Confusion Matrix:')
    print(results)

    print('Classification Report:')
    print(classification_report(y_test, labels))

    f1_score_val = f1_score(y_test, labels)
    print('F1 Score:', f1_score_val)

    sns.heatmap(results, annot=True, cmap="coolwarm", linewidth=0.2)

    with open('model_new.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

detect_spam()