from sklearn.ensemble import RandomForestClassifier
import langdetect
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns



def detect_spam_rf():
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

    random = TfidfVectorizer(lowercase=True, stop_words='english')
    features_train_transformed = random.fit_transform(x_train)
    features_test_transformed = random.transform(x_test)

    with open('random_new.pkl', 'wb') as random_file:
        pickle.dump(random, random_file)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(features_train_transformed, y_train)

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

    with open('model_new_rf.pkl', 'wb') as tree_file:
        pickle.dump(classifier, tree_file)

detect_spam_rf()
