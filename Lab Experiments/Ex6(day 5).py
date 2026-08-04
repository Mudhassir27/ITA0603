from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
X=[[1],[2],[3],[6],[7],[8]]
y=[0,0,0,1,1,1]
m=GaussianNB()
m.fit(X,y)
p=m.predict(X)
print(confusion_matrix(y,p))
print("Accuracy:",accuracy_score(y,p))
print("----------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
X = [[9.1, 90, 88],
    [8.8, 85, 84],
    [8.2, 82, 80],
    [7.0, 70, 68],
    [6.5, 65, 60],
    [9.3, 92, 90],
    [7.4, 72, 70],
    [8.7, 86, 83],
    [6.8, 67, 65],
    [9.0, 89, 87]]
y = ["Placed",
    "Placed",
    "Placed",
    "Not Placed",
    "Not Placed",
    "Placed",
    "Not Placed",
    "Placed",
    "Not Placed",
    "Placed"]
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred, labels=["Placed", "Not Placed"])
accuracy = accuracy_score(y, y_pred)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy * 100, "%")
print("-------------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
X = [[8, 780, 5],
    [7, 760, 4],
    [6, 730, 5],
    [4, 620, 8],
    [3, 590, 9],
    [9, 810, 4],
    [5, 650, 7],
    [8, 770, 5],
    [4, 610, 8],
    [7, 750, 6]]
y = ["Yes",
    "Yes",
    "Yes",
    "No",
    "No",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes"]
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred, labels=["Yes", "No"])
accuracy = accuracy_score(y, y_pred)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy * 100, "%")
print("-------------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
X = [[39.1, 112, 94],
    [38.8, 109, 95],
    [37.0, 78, 99],
    [39.3, 115, 93],
    [36.8, 74, 98],
    [38.7, 110, 94],
    [37.1, 80, 98],
    [39.0, 113, 92],
    [36.9, 76, 99],
    [38.9, 111, 93]]
y = ["Positive",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive"]
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred, labels=["Positive", "Negative"])
accuracy = accuracy_score(y, y_pred)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy * 100, "%")
print("-------------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
X = [[10, 95, 50],
    [9, 92, 48],
    [8, 90, 45],
    [4, 70, 25],
    [3, 65, 20],
    [11, 96, 52],
    [5, 75, 30],
    [8, 89, 44],
    [2, 60, 18],
    [7, 87, 40]]
y = ["Yes",
    "Yes",
    "Yes",
    "No",
    "No",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes"]
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred, labels=["Yes", "No"])
accuracy = accuracy_score(y, y_pred)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy * 100, "%")
print("--------------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
X = [[150, 7.5, 90],
    [160, 7.8, 88],
    [145, 7.3, 91],
    [120, 5.5, 70],
    [125, 5.8, 72],
    [130, 6.0, 74],
    [155, 7.6, 89],
    [118, 5.4, 69],
    [148, 7.4, 92],
    [122, 5.7, 71]]
y = ["Apple",
    "Apple",
    "Apple",
    "Orange",
    "Orange",
    "Orange",
    "Apple",
    "Orange",
    "Apple",
    "Orange"]
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred, labels=["Apple", "Orange"])
accuracy = accuracy_score(y, y_pred)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy * 100, "%")
