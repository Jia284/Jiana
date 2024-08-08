from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    )



X,y=make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    )
fig = plt.scatter(X[:,0],X[:,1],c=y,marker='*')
fig.figure.savefig('temp.png')

X_train,X_test, y_train,y_test=train_test_split(
    X,y,test_size=0.33, random_state=125)

model=GaussianNB()
model.fit(X_train, y_train)
predicted =model.predict([X_test[6]])

print("actual value:", y_test[6])
print("predicted value", predicted[0])

y_pred= model.predict(X_test)
accuracy=accuracy_score(y_pred, y_test)
f1=f1_score(y_pred, y_test, average='weighted')

print("accuracy: " , accuracy)
print("f1 score:", f1)


labels = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();

import pandas as pd


df = pd.read_csv('loan_data.csv')
df.head()
df.info()